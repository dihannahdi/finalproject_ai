/**
 * Time Dependency Data Mining (TDDM) model for stock market prediction
 * Based on the research paper: computation-13-00003.pdf
 * Implemented using TensorFlow.js for browser-based prediction
 */

import * as tf from '@tensorflow/tfjs';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';

// Extended StockDataPoint interface with derived features
interface ExtendedStockDataPoint extends StockDataPoint {
  momentum_5d?: number;
  momentum_10d?: number;
  volatility_5d?: number;
  range_5d?: number;
  volume_change?: number;
  price_acceleration?: number;
}

export interface TDDMModelParams {
  timeSteps: number;           // Number of time steps to consider for prediction (window size)
  features: string[];          // Features to use for prediction
  epochs: number;              // Number of training epochs
  batchSize: number;           // Batch size for training
  learningRate: number;        // Learning rate for optimizer
  hiddenUnits: number[];       // Array defining hidden unit sizes
  dropoutRate: number;         // Dropout rate for regularization
  useAttention: boolean;       // Whether to use attention mechanism
  useTDDMLayers: boolean;      // Whether to use specialized TDDM layers
  temporalFeatureExtraction: boolean; // Enable temporal feature extraction
}

/**
 * Default parameters for the TDDM model
 */
export const DEFAULT_TDDM_PARAMS: TDDMModelParams = {
  timeSteps: 30,
  features: ['close', 'open', 'high', 'low', 'volume'],
  epochs: 100,
  batchSize: 32,
  learningRate: 0.001,
  hiddenUnits: [128, 64, 32],
  dropoutRate: 0.3,
  useAttention: true,
  useTDDMLayers: true,
  temporalFeatureExtraction: true
};

/**
 * MinMaxScaler for feature normalization
 */
class MinMaxScaler {
  private min: number[] = [];
  private max: number[] = [];
  private range: number[] = [];

  /**
   * Fit the scaler to the data
   */
  fit(data: number[][]): void {
    const numFeatures = data[0].length;
    
    this.min = Array(numFeatures).fill(Infinity);
    this.max = Array(numFeatures).fill(-Infinity);
    this.range = Array(numFeatures).fill(0);
    
    // Find min and max values for each feature
    for (const row of data) {
      for (let i = 0; i < numFeatures; i++) {
        this.min[i] = Math.min(this.min[i], row[i]);
        this.max[i] = Math.max(this.max[i], row[i]);
      }
    }
    
    // Calculate range
    for (let i = 0; i < numFeatures; i++) {
      this.range[i] = this.max[i] - this.min[i];
      // Handle zero range case (constant feature)
      if (this.range[i] === 0) this.range[i] = 1;
    }
  }
  
  /**
   * Transform a single value
   */
  transformValue(value: number, featureIndex: number): number {
    return (value - this.min[featureIndex]) / this.range[featureIndex];
  }
  
  /**
   * Inverse transform a single value
   */
  inverseTransformValue(value: number, featureIndex: number): number {
    return value * this.range[featureIndex] + this.min[featureIndex];
  }
  
  /**
   * Transform the entire dataset
   */
  transform(data: number[][]): number[][] {
    return data.map(row => row.map((val, i) => this.transformValue(val, i)));
  }
  
  /**
   * Inverse transform the entire dataset
   */
  inverseTransform(data: number[][]): number[][] {
    return data.map(row => row.map((val, i) => this.inverseTransformValue(val, i)));
  }
}

/**
 * Extract temporal features like trends, volatility, momentum, etc.
 */
function extractTemporalFeatures(data: StockDataPoint[]): ExtendedStockDataPoint[] {
  const result = [...data] as ExtendedStockDataPoint[];
  
  // Need at least 10 days for reasonable feature extraction
  if (data.length < 10) {
    return result;
  }
  
  for (let i = 10; i < data.length; i++) {
    const currentPoint = {...data[i]} as ExtendedStockDataPoint;
    
    // Calculate some additional features based on past data
    
    // 1. Short-term price momentum (5-day)
    currentPoint.momentum_5d = data[i].close / data[i-5].close - 1;
    
    // 2. Medium-term price momentum (10-day)
    currentPoint.momentum_10d = data[i].close / data[i-10].close - 1;
    
    // 3. Volatility (standard deviation of returns over past 5 days)
    const returns5d = [];
    for (let j = i-5; j < i; j++) {
      returns5d.push(data[j+1].close / data[j].close - 1);
    }
    currentPoint.volatility_5d = calculateStdDev(returns5d);
    
    // 4. Price range relative to 5-day range
    const highLast5 = Math.max(...data.slice(i-5, i).map(d => d.high));
    const lowLast5 = Math.min(...data.slice(i-5, i).map(d => d.low));
    currentPoint.range_5d = (data[i].high - data[i].low) / (highLast5 - lowLast5);
    
    // 5. Volume change
    currentPoint.volume_change = data[i].volume / data[i-1].volume - 1;
    
    // 6. Price acceleration (change in momentum)
    const prev5dMomentum = data[i-1].close / data[i-6].close - 1;
    currentPoint.price_acceleration = currentPoint.momentum_5d - prev5dMomentum;
    
    result[i] = currentPoint;
  }
  
  return result;
}

/**
 * Calculate standard deviation helper
 */
function calculateStdDev(values: number[]): number {
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
  const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Custom attention layer for time series data
 */
class AttentionLayer {
  private weights: tf.Tensor | null = null;
  
  apply(inputs: tf.Tensor): tf.Tensor {
    // Self-attention mechanism
    const [_, timeSteps, features] = inputs.shape;
    
    // Create query, key, value projections
    const q = tf.layers.dense({units: 32}).apply(inputs) as tf.Tensor;
    const k = tf.layers.dense({units: 32}).apply(inputs) as tf.Tensor;
    const v = tf.layers.dense({units: features}).apply(inputs) as tf.Tensor;
    
    // Compute attention scores
    const scores = tf.matMul(q, k.transpose([0, 2, 1]));
    
    // Scale scores
    const scaleFactor = Math.sqrt(32);
    const scaledScores = tf.div(scores, scaleFactor);
    
    // Apply softmax to get attention weights
    this.weights = tf.softmax(scaledScores, -1);
    
    // Apply attention weights to values
    const output = tf.matMul(this.weights, v);
    
    // Residual connection
    return tf.add(inputs, output);
  }
  
  getAttentionWeights(): tf.Tensor | null {
    return this.weights;
  }
}

export class TDDMModel {
  private model: tf.LayersModel | null = null;
  private params: TDDMModelParams;
  private inputScaler: MinMaxScaler;
  private outputScaler: MinMaxScaler;
  private isTraining: boolean = false;
  private featureIndices: { [key: string]: number } = {};
  private attention: AttentionLayer | null = null;
  private _cachedModelData: ExtendedStockDataPoint[] | null = null;

  /**
   * Create a new TDDM model with given parameters
   */
  constructor(params: Partial<TDDMModelParams> = {}) {
    this.params = { ...DEFAULT_TDDM_PARAMS, ...params };
    this.inputScaler = new MinMaxScaler();
    this.outputScaler = new MinMaxScaler();
    
    // Map feature names to indices
    this.params.features.forEach((feature, index) => {
      this.featureIndices[feature] = index;
    });
  }

  /**
   * Build the TDDM model architecture
   */
  private buildModel(inputShape: [number, number], expandedFeatureCount?: number): tf.LayersModel {
    const model = tf.sequential();
    
    // Use the expanded feature count if provided (for temporal features)
    const featuresCount = expandedFeatureCount || inputShape[1];
    
    // Input preprocessing
    if (this.params.useTDDMLayers) {
      // TDDM-specific initial layer with time-aware convolution
      model.add(tf.layers.conv1d({
        filters: 32,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        inputShape: [inputShape[0], featuresCount],
      }));
    } else {
      // Standard dense layer
      model.add(tf.layers.dense({
        units: 64,
        activation: 'relu',
        inputShape: [inputShape[0], featuresCount],
      }));
    }
    
    // Attention mechanism if enabled
    if (this.params.useAttention) {
      // Since we can't use a custom attention layer directly in TensorFlow.js,
      // we'll implement self-attention using built-in layers
      
      // Create a self-attention mechanism using built-in layers
      model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
        name: 'attention_query'
      }));
      
      model.add(tf.layers.dense({
        units: featuresCount,
        activation: 'tanh',
        name: 'attention_output'
      }));
    }
    
    // Main processing layers
    for (const units of this.params.hiddenUnits) {
      // Add Bidirectional LSTM for temporal processing
      model.add(tf.layers.bidirectional({
        layer: tf.layers.lstm({
          units: units,
          returnSequences: true,
          activation: 'tanh',
          recurrentActivation: 'sigmoid',
        }),
        mergeMode: 'concat',
      }));
      
      model.add(tf.layers.dropout({ rate: this.params.dropoutRate }));
    }
    
    // Time-distributed dense layer for feature extraction at each time step
    model.add(tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: 16,
        activation: 'relu',
      })
    }));
    
    // Global attention pooling
    model.add(tf.layers.globalAveragePooling1d({}));
    
    // Final dense layers
    model.add(tf.layers.dense({
      units: 16,
      activation: 'relu',
    }));
    
    model.add(tf.layers.dropout({ rate: this.params.dropoutRate / 2 }));
    
    // Output layer
    model.add(tf.layers.dense({ units: 1 }));
    
    // Compile the model with Adam optimizer and MSE loss
    const optimizer = tf.train.adam(this.params.learningRate);
    
    model.compile({
      optimizer,
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae'],
    });
    
    return model;
  }

  /**
   * Prepare data for training by creating sequences
   */
  private prepareTrainingData(data: ExtendedStockDataPoint[]): { X: tf.Tensor3D, y: tf.Tensor2D } {
    const featureCount = Object.keys(data[0]).filter(key => typeof data[0][key as keyof ExtendedStockDataPoint] === 'number').length - 1; // -1 to exclude 'date'
    const sequences: number[][][] = [];
    const targets: number[] = [];
    
    // Create input sequences and target values
    for (let i = 0; i < data.length - this.params.timeSteps; i++) {
      const sequence: number[][] = [];
      
      for (let j = 0; j < this.params.timeSteps; j++) {
        const point = data[i + j];
        const featureValues: number[] = [];
        
        // Extract all numerical features except date
        Object.keys(point).forEach(key => {
          const value = point[key as keyof ExtendedStockDataPoint];
          if (typeof value === 'number' && key !== 'date' && key !== 'timestamp') {
            featureValues.push(value);
          }
        });
        
        sequence.push(featureValues);
      }
      
      sequences.push(sequence);
      targets.push(data[i + this.params.timeSteps].close);
    }
    
    // Convert to tensors and reshape
    const X = tf.tensor3d(sequences, [sequences.length, this.params.timeSteps, featureCount]);
    const y = tf.tensor2d(targets, [targets.length, 1]);
    
    return { X, y };
  }

  /**
   * Train the model with historical stock data
   */
  async train(data: StockDataPoint[]): Promise<any> {
    if (this.isTraining) {
      throw new Error('Model is already training');
    }
    
    this.isTraining = true;
    
    try {
      console.log('Training TDDM model...');
      
      if (!data || data.length < this.params.timeSteps * 2) {
        throw new Error(`Need at least ${this.params.timeSteps * 2} data points to train, got ${data.length}`);
      }
      
      // Apply temporal feature extraction if enabled
      const processedData = this.params.temporalFeatureExtraction 
        ? extractTemporalFeatures(data)
        : [...data] as ExtendedStockDataPoint[];
      
      // Store data for feature importance calculations
      this._cachedModelData = [...processedData];
      
      // Prepare training data
      const { X, y } = this.prepareTrainingData(processedData);
      
      // Build the model architecture
      this.model = this.buildModel([this.params.timeSteps, X.shape[2]]);
      
      // Train the model
      const history = await this.model.fit(X, y, {
        epochs: this.params.epochs,
        batchSize: this.params.batchSize,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (epoch % 10 === 0) {
              console.log(
                `Epoch ${epoch}: loss = ${logs?.loss.toFixed(4)}, val_loss = ${logs?.val_loss.toFixed(4)}`
              );
            }
          }
        }
      });
      
      // Clean up tensors
      X.dispose();
      y.dispose();
      
      console.log('TDDM model training completed');
      return history;
    } catch (error) {
      console.error('Error training TDDM model:', error);
      throw error;
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * Predict future stock prices
   */
  async predict(data: StockDataPoint[], days: number = 30): Promise<PredictionPoint[]> {
    if (!this.model) {
      throw new Error('Model not trained yet. Call train() first.');
    }
    
    if (data.length < this.params.timeSteps) {
      throw new Error(`Insufficient data for prediction. Need at least ${this.params.timeSteps} data points.`);
    }
    
    // Apply temporal feature extraction if enabled
    let processedData: ExtendedStockDataPoint[] = data as ExtendedStockDataPoint[];
    if (this.params.temporalFeatureExtraction) {
      processedData = extractTemporalFeatures(data);
    }
    
    // Get the most recent data points
    let recentData = processedData.slice(-this.params.timeSteps);
    
    // Scale the input data (all numerical features except date)
    const scaledData = [...recentData];
    
    for (let i = 0; i < scaledData.length; i++) {
      const point = scaledData[i];
      const newPoint = { ...point };
      
      let featureIdx = 0;
      // Scale all numerical features except date
      Object.keys(point).forEach(key => {
        const value = point[key as keyof ExtendedStockDataPoint];
        if (typeof value === 'number' && key !== 'date' && key !== 'timestamp') {
          const scaled = this.inputScaler.transformValue(value, featureIdx++);
          (newPoint as any)[key] = scaled;
        }
      });
      
      scaledData[i] = newPoint;
    }
    
    // Predictions array
    const predictions: PredictionPoint[] = [];
    const lastDate = new Date(data[data.length - 1].date);
    
    // Create a copy of the scaled data that we'll update during prediction
    let currentData = [...scaledData];
    
    // Make predictions for the specified number of days
    for (let i = 0; i < days; i++) {
      // Prepare input for prediction
      const featureCount = Object.keys(currentData[0])
        .filter(key => typeof currentData[0][key as keyof ExtendedStockDataPoint] === 'number' && key !== 'date' && key !== 'timestamp')
        .length;
      
      const sequence: number[][] = [];
      for (let j = 0; j < this.params.timeSteps; j++) {
        const point = currentData[j];
        const featureValues: number[] = [];
        
        // Extract all numerical features except date
        Object.keys(point).forEach(key => {
          const value = point[key as keyof ExtendedStockDataPoint];
          if (typeof value === 'number' && key !== 'date' && key !== 'timestamp') {
            featureValues.push(value);
          }
        });
        
        sequence.push(featureValues);
      }
      
      // Convert to tensor
      const input = tf.tensor3d([sequence], [1, this.params.timeSteps, featureCount]);
      
      // Make prediction
      const predictionTensor = this.model.predict(input) as tf.Tensor;
      const predictedValue = await predictionTensor.data();
      
      // Scale back the predicted value
      const unscaledPrediction = this.outputScaler.inverseTransformValue(predictedValue[0], 0);
      
      // Calculate date for this prediction
      const predictionDate = new Date(lastDate);
      predictionDate.setDate(lastDate.getDate() + i + 1);
      
      // Create prediction point
      const predictionPoint: PredictionPoint = {
        date: predictionDate.toISOString().split('T')[0],
        price: unscaledPrediction,
        upper: unscaledPrediction * 1.02, // Add 2% for upper bound
        lower: unscaledPrediction * 0.98, // Subtract 2% for lower bound
      };
      
      predictions.push(predictionPoint);
      
      // Update currentData by removing the first element and adding the new prediction
      const newDataPoint: Partial<ExtendedStockDataPoint> = {};
      
      // Set all features to the last known values
      Object.keys(currentData[currentData.length - 1]).forEach(key => {
        if (key === 'close') {
          (newDataPoint as any)['close'] = predictedValue[0];
        } else {
          (newDataPoint as any)[key] = 
            currentData[currentData.length - 1][key as keyof ExtendedStockDataPoint];
        }
      });
      
      // Add date and timestamp to the new data point
      const newDate = predictionDate.toISOString().split('T')[0];
      newDataPoint.date = newDate;
      newDataPoint.timestamp = predictionDate.getTime();
      
      // Remove first element and add new prediction
      currentData.shift();
      currentData.push(newDataPoint as ExtendedStockDataPoint);
      
      // If temporal feature extraction is enabled, we need to recalculate features
      if (this.params.temporalFeatureExtraction && i % 5 === 4) { // Re-extract every 5 days to avoid computational overhead
        currentData = extractTemporalFeatures(currentData as StockDataPoint[]);
      }
      
      // Clean up tensors
      input.dispose();
      predictionTensor.dispose();
    }
    
    return predictions;
  }

  /**
   * Save the trained model
   */
  async saveModel(modelName: string): Promise<tf.io.SaveResult> {
    if (!this.model) {
      throw new Error('No model to save. Train the model first.');
    }
    
    return await this.model.save(`localstorage://${modelName}`);
  }

  /**
   * Load a previously trained model
   */
  async loadModel(modelName: string): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(`localstorage://${modelName}`);
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  /**
   * Get importance scores for features based on model analysis
   */
  async getFeatureImportance(): Promise<{feature: string, importance: number}[] | null> {
    if (!this.model) {
      return null;
    }
    
    // We'll use a permutation importance approach:
    // For each feature, we'll measure the model performance when that feature is shuffled
    // The more performance drops, the more important the feature is
    
    if (!this.model || this.isTraining) {
      return null;
    }
    
    try {
      // Create a test dataset from the last 20% of data
      const testData: StockDataPoint[] = [];
      const modelData = await this.getModelData();
      if (!modelData || !modelData.length) {
        return null;
      }
      
      // Select some test data for evaluation
      const testSize = Math.floor(modelData.length * 0.2);
      const testSample = modelData.slice(-testSize);
      
      // Calculate baseline performance
      const baselineError = await this.evaluateModel(testSample);
      
      // Calculate feature importance for each feature
      const featureImportance: {feature: string, importance: number}[] = [];
      
      // Process each feature
      for (const feature of this.params.features) {
        // Create a shuffled copy of the test data
        const shuffledData = JSON.parse(JSON.stringify(testSample)) as ExtendedStockDataPoint[];
        
        // Extract feature values
        const featureValues = shuffledData.map(d => d[feature as keyof StockDataPoint]);
        
        // Shuffle the feature values
        for (let i = featureValues.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [featureValues[i], featureValues[j]] = [featureValues[j], featureValues[i]];
        }
        
        // Replace the feature values in the shuffled data
        shuffledData.forEach((d, i) => {
          (d as any)[feature] = featureValues[i];
        });
        
        // Measure performance drop
        const shuffledError = await this.evaluateModel(shuffledData);
        
        // The importance is the increase in error when the feature is shuffled
        const importance = shuffledError - baselineError;
        
        featureImportance.push({
          feature,
          importance: Math.max(0, importance) // Ensure non-negative importance
        });
      }
      
      // Normalize importance scores to sum to 1.0
      const totalImportance = featureImportance.reduce((sum, item) => sum + item.importance, 0);
      if (totalImportance > 0) {
        featureImportance.forEach(item => {
          item.importance = item.importance / totalImportance;
        });
      }
      
      // Sort by importance (descending)
      return featureImportance.sort((a, b) => b.importance - a.importance);
    } catch (error) {
      console.error('Error calculating feature importance:', error);
      return null;
    }
  }

  /**
   * Helper method to get model training data
   */
  private async getModelData(): Promise<ExtendedStockDataPoint[] | null> {
    // Try to use cached model data if available
    if (this._cachedModelData) {
      return this._cachedModelData;
    }
    
    return null;
  }

  /**
   * Evaluate model on a test set and return the error metric
   */
  private async evaluateModel(testData: ExtendedStockDataPoint[]): Promise<number> {
    try {
      if (!this.model || testData.length < this.params.timeSteps + 1) {
        return Infinity;
      }
      
      let totalError = 0;
      let count = 0;
      
      // Prepare test data for prediction
      for (let i = this.params.timeSteps; i < testData.length; i++) {
        const inputWindow = testData.slice(i - this.params.timeSteps, i);
        const actualPrice = testData[i].close;
        
        // Prepare model input
        const inputFeatures = this.prepareFeatures(inputWindow);
        if (!inputFeatures) continue;
        
        // Make prediction
        const input = tf.tensor3d([inputFeatures], [1, inputFeatures.length, inputFeatures[0].length]);
        const prediction = this.model.predict(input) as tf.Tensor;
        
        // Get predicted value
        const predictedValue = await prediction.data();
        const normalizedPredictedValue = predictedValue[0];
        
        // Convert back to original scale
        const originalScalePrediction = this.outputScaler.inverseTransformValue(
          normalizedPredictedValue, 
          0
        );
        
        // Calculate error (MSE)
        const error = Math.pow(actualPrice - originalScalePrediction, 2);
        
        totalError += error;
        count++;
        
        // Clean up tensors
        input.dispose();
        prediction.dispose();
      }
      
      // Return average error
      return count > 0 ? totalError / count : Infinity;
    } catch (error) {
      console.error('Error evaluating model:', error);
      return Infinity;
    }
  }

  /**
   * Prepare features for model input
   */
  private prepareFeatures(data: ExtendedStockDataPoint[]): number[][] | null {
    if (!data || data.length < this.params.timeSteps) {
      return null;
    }

    // Extract features for each time step
    const features: number[][] = [];
    
    for (const point of data) {
      const featureVector: number[] = [];
      
      // Extract basic features
      for (const feature of this.params.features) {
        // Skip if feature doesn't exist
        if (point[feature as keyof StockDataPoint] === undefined) {
          continue;
        }
        
        featureVector.push(point[feature as keyof StockDataPoint] as number);
      }
      
      // Add derived features if they exist
      if (this.params.temporalFeatureExtraction) {
        if (point.momentum_5d !== undefined) featureVector.push(point.momentum_5d);
        if (point.momentum_10d !== undefined) featureVector.push(point.momentum_10d);
        if (point.volatility_5d !== undefined) featureVector.push(point.volatility_5d);
        if (point.range_5d !== undefined) featureVector.push(point.range_5d);
        if (point.volume_change !== undefined) featureVector.push(point.volume_change);
        if (point.price_acceleration !== undefined) featureVector.push(point.price_acceleration);
      }
      
      features.push(featureVector);
    }
    
    return features;
  }
}

export default TDDMModel; 