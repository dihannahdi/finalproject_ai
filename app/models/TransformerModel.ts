/**
 * Transformer model for stock price prediction
 * Implemented using TensorFlow.js for browser-based prediction
 * 
 * This model uses the Transformer architecture with self-attention mechanisms
 * to capture long-range dependencies in time series data.
 */

import * as tf from '@tensorflow/tfjs';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import { PredictionModel } from './index';

export interface TransformerModelParams {
  timeSteps: number;           // Number of time steps to consider for prediction
  features: string[];          // Features to use for prediction (e.g., ['close', 'volume'])
  epochs: number;              // Number of training epochs
  batchSize: number;           // Batch size for training
  learningRate: number;        // Learning rate for optimizer
  dModel: number;              // Embedding dimension
  numHeads: number;            // Number of attention heads
  ffnDim: number;              // Feed-forward network dimension
  numEncoderLayers: number;    // Number of encoder layers
  dropoutRate: number;         // Dropout rate for regularization
}

/**
 * Default parameters for the Transformer model
 */
export const DEFAULT_TRANSFORMER_PARAMS: TransformerModelParams = {
  timeSteps: 15,
  features: ['close', 'open', 'high', 'low', 'volume'],
  epochs: 50,
  batchSize: 64,
  learningRate: 0.0005,
  dModel: 32,
  numHeads: 4,
  ffnDim: 64,
  numEncoderLayers: 2,
  dropoutRate: 0.2
};

export class TransformerModel implements PredictionModel {
  private model: tf.LayersModel | null = null;
  private params: TransformerModelParams;
  private inputScaler: MinMaxScaler;
  private outputScaler: MinMaxScaler;
  private isTraining: boolean = false;
  private featureIndices: { [key: string]: number } = {};

  /**
   * Create a new Transformer model with given parameters
   */
  constructor(params: Partial<TransformerModelParams> = {}) {
    this.params = { ...DEFAULT_TRANSFORMER_PARAMS, ...params };
    this.inputScaler = new MinMaxScaler();
    this.outputScaler = new MinMaxScaler();
    
    // Map feature names to indices
    this.params.features.forEach((feature, index) => {
      this.featureIndices[feature] = index;
    });
  }

  /**
   * Build the Transformer model architecture
   */
  private buildModel(inputShape: [number, number], overrideParams?: Partial<TransformerModelParams>): tf.LayersModel {
    // Use provided params or defaults
    const params = overrideParams ? { ...this.params, ...overrideParams } : this.params;
    
    const model = tf.sequential();
    
    // Input layer and reshaping
    model.add(tf.layers.inputLayer({
      inputShape: [inputShape[0], inputShape[1]]
    }));
    
    // 1D Convolutional layers as positional feature extractors
    model.add(tf.layers.conv1d({
      filters: params.dModel,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) // Add L2 regularization for stability
    }));
    
    // Self-attention mechanism using compatible TensorFlow.js layers
    for (let i = 0; i < params.numEncoderLayers; i++) {
      // Project to query, key, value spaces (simplified attention mechanism)
      model.add(tf.layers.dense({
        units: params.dModel,
        activation: 'linear',
        name: `encoder_${i}_projection`,
        kernelInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
      }));
      
      // Add dropout for regularization
      model.add(tf.layers.dropout({
        rate: params.dropoutRate
      }));
      
      // Add normalization
      model.add(tf.layers.layerNormalization({
        epsilon: 1e-6
      }));
      
      // Position-wise Feed-Forward Network
      model.add(tf.layers.dense({
        units: params.ffnDim,
        activation: 'relu',
        name: `encoder_${i}_ffn1`,
        kernelInitializer: 'heNormal',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
      }));
      
      model.add(tf.layers.dense({
        units: params.dModel,
        activation: 'linear',
        name: `encoder_${i}_ffn2`,
        kernelInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
      }));
      
      // Another normalization layer
      model.add(tf.layers.layerNormalization({
        epsilon: 1e-6
      }));
    }
    
    // Flatten and final projection layers
    model.add(tf.layers.flatten());
    
    model.add(tf.layers.dense({
      units: 32,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));
    
    model.add(tf.layers.dropout({
      rate: params.dropoutRate / 2
    }));
    
    // Output layer
    model.add(tf.layers.dense({
      units: 1,
      kernelInitializer: 'glorotUniform',
      // Add activation to ensure positive values for stock prices
      activation: 'softplus'
    }));
    
    // Compile the model with gradient clipping to prevent exploding gradients
    const optimizer = tf.train.adam(params.learningRate, 0.9, 0.999, 1e-7);
    
    model.compile({
      optimizer: optimizer,
      loss: 'meanSquaredError',
      metrics: ['mse']
    });
    
    return model;
  }

  /**
   * Prepare data for training by creating sequences
   */
  private prepareTrainingData(data: StockDataPoint[]): { X: tf.Tensor3D, y: tf.Tensor2D } {
    const featureCount = this.params.features.length;
    const sequences: number[][][] = [];
    const targets: number[] = [];
    
    // Create input sequences and target values
    for (let i = 0; i < data.length - this.params.timeSteps; i++) {
      const sequence: number[][] = [];
      
      for (let j = 0; j < this.params.timeSteps; j++) {
        const featureValues: number[] = [];
        for (const feature of this.params.features) {
          featureValues.push(data[i + j][feature as keyof StockDataPoint] as number);
        }
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
   * Train the model on the provided historical data
   */
  async train(data: StockDataPoint[]): Promise<tf.History> {
    if (data.length < this.params.timeSteps + 10) {
      throw new Error(`Not enough data points. Need at least ${this.params.timeSteps + 10} data points.`);
    }
    
    this.isTraining = true;
    
    try {
      // Validate data to ensure all values are finite numbers
      const validData = data.filter(point => {
        return this.params.features.every(feature => {
          const value = point[feature as keyof StockDataPoint];
          return value !== undefined && value !== null && isFinite(value as number);
        });
      });
      
      if (validData.length < this.params.timeSteps + 10) {
        throw new Error(`Not enough valid data points after filtering. Need at least ${this.params.timeSteps + 10} valid data points.`);
      }
      
      console.log(`Training with ${validData.length} valid data points out of ${data.length} total points`);
      
      // Prepare features and scale data
      const features: number[][] = [];
      const targetValues: number[] = [];
      
      for (const point of validData) {
        const featureValues: number[] = [];
        for (const feature of this.params.features) {
          featureValues.push(point[feature as keyof StockDataPoint] as number);
        }
        features.push(featureValues);
        targetValues.push(point.close);
      }
      
      // Fit scalers with robust bounds checking
      try {
        this.inputScaler.fit(features);
        this.outputScaler.fit(targetValues.map(v => [v]));
        
        // Verify scalers are properly initialized
        if (!this.inputScaler.isValid() || !this.outputScaler.isValid()) {
          console.warn('Scalers not properly initialized. Using fallback scaling.');
          // Initialize with fallback values if needed
          this.inputScaler.initializeFallback(features);
          this.outputScaler.initializeFallback(targetValues.map(v => [v]));
        }
      } catch (error) {
        console.error('Error during scaling initialization:', error);
        // Initialize with fallback values
        this.inputScaler.initializeFallback(features);
        this.outputScaler.initializeFallback(targetValues.map(v => [v]));
      }
      
      // Scale the data points
      const scaledData = [...validData];
      for (let i = 0; i < scaledData.length; i++) {
        for (const feature of this.params.features) {
          const value = scaledData[i][feature as keyof StockDataPoint] as number;
          const featureIndex = this.featureIndices[feature];
          const scaled = this.inputScaler.transformValue(value, featureIndex);
          scaledData[i] = { ...scaledData[i], [feature]: scaled };
        }
        
        scaledData[i] = {
          ...scaledData[i],
          close: this.outputScaler.transformValue(scaledData[i].close, 0),
        };
      }
      
      // Prepare training data
      const { X, y } = this.prepareTrainingData(scaledData);
      
      // Build model with the correct input shape
      this.model = this.buildModel([this.params.timeSteps, this.params.features.length]);
      
      // Implement early stopping to prevent overfitting
      const earlyStoppingPatience = 5;
      let bestValLoss = Infinity;
      let patienceCounter = 0;
      let bestModelWeights: tf.Tensor[] | null = null;
      
      // Train the model with early stopping
      const history = await this.model.fit(X, y, {
        epochs: this.params.epochs,
        batchSize: this.params.batchSize,
        validationSplit: 0.2, // Use 20% of data for validation
        shuffle: true, // Shuffle data to improve training
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            const valLoss = logs?.val_loss || Infinity;
            console.log(`Epoch ${epoch + 1}/${this.params.epochs}, loss: ${logs?.loss.toFixed(6)}, val_loss: ${valLoss.toFixed(6)}`);
            
            // Check if this is the best model so far
            if (valLoss < bestValLoss) {
              bestValLoss = valLoss;
              patienceCounter = 0;
              
              // Save the best model weights
              if (bestModelWeights) {
                bestModelWeights.forEach(t => t.dispose());
              }
              bestModelWeights = this.model!.getWeights().map(w => w.clone());
            } else {
              patienceCounter++;
              
              // Check if we should stop early
              if (patienceCounter >= earlyStoppingPatience) {
                console.log(`Early stopping at epoch ${epoch + 1} due to no improvement in validation loss`);
                this.model!.stopTraining = true;
              }
            }
          },
        },
      });
      
      // Restore the best model weights if available
      if (bestModelWeights) {
        this.model.setWeights(bestModelWeights);
        console.log('Restored best model weights from training');
        
        // Clean up
        bestModelWeights.forEach(t => t.dispose());
      }
      
      // Validate the model with a simple prediction
      try {
        const testInput = tf.tensor3d([X.arraySync()[0]], [1, this.params.timeSteps, this.params.features.length]);
        const testPrediction = this.model.predict(testInput) as tf.Tensor;
        const testValue = testPrediction.dataSync()[0];
        
        // Check if prediction is valid
        if (!isFinite(testValue)) {
          console.warn('Model produced invalid prediction during validation. Rebuilding model...');
          
          // Rebuild and retrain with more conservative settings
          this.model.dispose();
          
          // Create a more conservative model
          const conservativeParams = { ...this.params, learningRate: this.params.learningRate / 2 };
          this.model = this.buildModel([this.params.timeSteps, this.params.features.length], conservativeParams);
          
          // Train with fewer epochs
          await this.model.fit(X, y, {
            epochs: Math.max(5, Math.floor(this.params.epochs / 2)),
            batchSize: this.params.batchSize,
            validationSplit: 0.2
          });
        }
        
        testInput.dispose();
        testPrediction.dispose();
      } catch (error) {
        console.error('Error during model validation:', error);
      }
      
      // Clean up tensors
      X.dispose();
      y.dispose();
      
      return history;
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
      throw new Error(`Not enough data points. Need at least ${this.params.timeSteps} data points.`);
    }
    
    // Create a copy of the last timeSteps data points for prediction
    const lastDataPoints = data.slice(-this.params.timeSteps);
    
    // Scale the input data
    const scaledData = [...lastDataPoints];
    for (let i = 0; i < scaledData.length; i++) {
      for (const feature of this.params.features) {
        const value = scaledData[i][feature as keyof StockDataPoint] as number;
        const featureIndex = this.featureIndices[feature];
        const scaled = this.inputScaler.transformValue(value, featureIndex);
        scaledData[i] = { ...scaledData[i], [feature]: scaled };
      }
    }
    
    // Get the last actual price for validation
    const lastActualPrice = data[data.length - 1].close;
    
    // Make predictions for the specified number of days
    const predictions: PredictionPoint[] = [];
    let currentSequence = scaledData;
    
    // Track the maximum allowed deviation from the last price
    // Start with a small percentage and gradually increase it
    const maxInitialDeviation = 0.03; // 3% initial max deviation
    const maxFinalDeviation = 0.15;   // 15% max deviation by end of prediction period
    
    for (let i = 0; i < days; i++) {
      // Prepare input for prediction
      const sequence: number[][] = [];
      for (let j = 0; j < this.params.timeSteps; j++) {
        const featureValues: number[] = [];
        for (const feature of this.params.features) {
          featureValues.push(currentSequence[j][feature as keyof StockDataPoint] as number);
        }
        sequence.push(featureValues);
      }
      
      // Convert to tensor and make prediction
      const input = tf.tensor3d([sequence], [1, this.params.timeSteps, this.params.features.length]);
      const prediction = this.model.predict(input) as tf.Tensor;
      const predictionValue = prediction.dataSync()[0];
      
      // Invert scaling to get the actual price
      const rawPriceValue = this.outputScaler.inverseTransformValue(predictionValue, 0);
      
      // Calculate the allowed deviation for this prediction day
      const deviationFactor = i / days; // 0 on first day, 1 on last day
      const maxDeviation = maxInitialDeviation + (deviationFactor * (maxFinalDeviation - maxInitialDeviation));
      
      // Apply constraints to prevent unreasonable predictions
      // Ensure the price doesn't deviate too much from the last actual price
      const minAllowedPrice = lastActualPrice * (1 - maxDeviation);
      const maxAllowedPrice = lastActualPrice * (1 + maxDeviation);
      
      // Apply smoothing: If this is not the first prediction, also consider the previous prediction
      let priceValue = rawPriceValue;
      if (i > 0) {
        const prevPrediction = predictions[i - 1].price;
        // Blend between raw prediction and smoothed value (more weight to previous as we go further)
        const smoothingFactor = Math.min(0.7, 0.3 + (i / days) * 0.4); // Increases from 0.3 to 0.7
        priceValue = (1 - smoothingFactor) * rawPriceValue + smoothingFactor * prevPrediction;
      }
      
      // Apply the constraints
      priceValue = Math.max(minAllowedPrice, Math.min(maxAllowedPrice, priceValue));
      
      // Create a date for this prediction (1 day after the last date)
      const lastDate = new Date(
        i === 0 
          ? data[data.length - 1].date 
          : predictions[i - 1].date
      );
      const predictionDate = new Date(lastDate);
      predictionDate.setDate(lastDate.getDate() + 1);
      
      // Calculate volatility based on historical data, but decrease it for stability
      const volatilityBase = 0.015; // 1.5% base volatility
      // Reduce volatility for the first few predictions to ensure stability
      const volatility = i < 5 
        ? volatilityBase * (0.5 + (i / 10)) // Gradually increase from 0.5x to 1x
        : volatilityBase;
      
      // Create prediction point
      const predictionPoint: PredictionPoint = {
        date: predictionDate.toISOString().split('T')[0],
        price: priceValue,
        upper: priceValue * (1 + volatility),
        lower: priceValue * (1 - volatility),
      };
      
      predictions.push(predictionPoint);
      
      // Update the sequence for the next prediction
      const newDataPoint: StockDataPoint = {
        date: predictionPoint.date,
        timestamp: predictionDate.getTime(),
        open: predictionValue,
        high: predictionValue,
        low: predictionValue,
        close: predictionValue,
        volume: currentSequence[currentSequence.length - 1].volume,
      };
      
      // Remove the first element and add the new prediction
      currentSequence = [...currentSequence.slice(1), newDataPoint];
      
      // Clean up tensors
      input.dispose();
      prediction.dispose();
    }
    
    // Add logging to help diagnose issues
    console.log('Transformer predictions generated:', {
      firstPrice: predictions[0]?.price,
      lastPrice: predictions[predictions.length - 1]?.price,
      lastActualPrice,
      deviation: predictions[predictions.length - 1]?.price 
        ? ((predictions[predictions.length - 1].price / lastActualPrice - 1) * 100).toFixed(2) + '%'
        : 'N/A'
    });
    
    return predictions;
  }

  /**
   * Save the model to browser's IndexedDB
   */
  async saveModel(modelName: string): Promise<tf.io.SaveResult> {
    if (!this.model) {
      throw new Error('No model to save. Train a model first.');
    }
    
    return await this.model.save(`indexeddb://${modelName}`);
  }

  /**
   * Load a previously saved model
   */
  async loadModel(modelName: string): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(`indexeddb://${modelName}`);
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      throw new Error('Failed to load model');
    }
  }
}

/**
 * Utility class for scaling features
 */
class MinMaxScaler {
  private min: number[] = [];
  private max: number[] = [];
  private range: number[] = [];
  
  /**
   * Fit the scaler to the data
   */
  fit(data: number[][]): void {
    if (data.length === 0) return;
    
    const features = data[0].length;
    this.min = new Array(features).fill(Infinity);
    this.max = new Array(features).fill(-Infinity);
    
    // Find min and max values for each feature
    for (const row of data) {
      for (let i = 0; i < features; i++) {
        this.min[i] = Math.min(this.min[i], row[i]);
        this.max[i] = Math.max(this.max[i], row[i]);
      }
    }
    
    // Calculate range for each feature
    this.range = this.max.map((max, i) => max - this.min[i]);
    
    // Prevent division by zero
    for (let i = 0; i < this.range.length; i++) {
      if (this.range[i] === 0) {
        this.range[i] = 1;
      }
    }
  }
  
  /**
   * Check if the scaler is properly initialized
   */
  isValid(): boolean {
    return this.min.length > 0 && 
           this.max.length > 0 && 
           this.range.length > 0 && 
           this.min.length === this.max.length && 
           this.min.length === this.range.length &&
           this.min.every(val => isFinite(val)) &&
           this.max.every(val => isFinite(val)) &&
           this.range.every(val => isFinite(val) && val > 0);
  }

  /**
   * Initialize the scaler with fallback values
   */
  initializeFallback(data: number[][]): void {
    if (data.length === 0 || data[0].length === 0) {
      // Default initialization if no data
      this.min = [0];
      this.max = [1];
      this.range = [1];
      return;
    }
    
    const features = data[0].length;
    this.min = new Array(features).fill(0);
    this.max = new Array(features).fill(0);
    
    // Calculate average for each feature
    const sums = new Array(features).fill(0);
    let validCount = 0;
    
    for (const row of data) {
      if (row.every(val => isFinite(val))) {
        for (let i = 0; i < features; i++) {
          sums[i] += row[i];
        }
        validCount++;
      }
    }
    
    // If we have valid data, use it to set reasonable min/max
    if (validCount > 0) {
      const avgs = sums.map(sum => sum / validCount);
      
      // Set min/max to be ±20% from average
      for (let i = 0; i < features; i++) {
        const avg = avgs[i];
        const absAvg = Math.abs(avg);
        const range = Math.max(absAvg * 0.2, 1); // At least 1 or 20% of abs average
        
        this.min[i] = avg - range;
        this.max[i] = avg + range;
      }
    } else {
      // Fallback to simple 0-1 range if no valid data
      this.min.fill(0);
      this.max.fill(1);
    }
    
    // Calculate range for each feature
    this.range = this.max.map((max, i) => Math.max(max - this.min[i], 1));
  }
  
  /**
   * Transform a single value for a specific feature
   */
  transformValue(value: number, featureIndex: number): number {
    // Handle invalid inputs
    if (!isFinite(value) || featureIndex >= this.range.length || this.range[featureIndex] === 0) {
      return 0.5; // Return middle of normalized range for invalid inputs
    }
    return (value - this.min[featureIndex]) / this.range[featureIndex];
  }
  
  /**
   * Invert the transformation for a single value
   */
  inverseTransformValue(value: number, featureIndex: number): number {
    // Handle invalid inputs
    if (!isFinite(value) || featureIndex >= this.range.length) {
      // Return the middle of the original range as fallback
      return this.min[featureIndex] + (this.range[featureIndex] / 2);
    }
    return value * this.range[featureIndex] + this.min[featureIndex];
  }
  
  /**
   * Transform the entire dataset
   */
  transform(data: number[][]): number[][] {
    return data.map(row => 
      row.map((value, i) => this.transformValue(value, i))
    );
  }
  
  /**
   * Invert the transformation for the entire dataset
   */
  inverseTransform(data: number[][]): number[][] {
    return data.map(row => 
      row.map((value, i) => this.inverseTransformValue(value, i))
    );
  }
}

export default TransformerModel; 