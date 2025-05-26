/**
 * CNN-LSTM (Convolutional Neural Network + Long Short-Term Memory) model for stock price prediction
 * Based on the research paper: Comparative Study on Stock Market Prediction using Generic CNN-LSTM and Ensemble Learning
 * Implemented using TensorFlow.js for browser-based prediction
 */

import * as tf from '@tensorflow/tfjs';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';

export interface CNNLSTMModelParams {
  timeSteps: number;           // Number of time steps to consider for prediction
  features: string[];          // Features to use for prediction (e.g., ['close', 'volume', 'open', 'high', 'low'])
  epochs: number;              // Number of training epochs
  batchSize: number;           // Batch size for training
  learningRate: number;        // Learning rate for optimizer
  cnnFilters: number;          // Number of CNN filters
  cnnKernelSize: number;       // CNN kernel size
  lstmUnits: number[];         // Array defining LSTM layer sizes
  dropoutRate: number;         // Dropout rate for regularization
  denseUnits: number[];        // Array defining dense layer sizes
}

/**
 * Default parameters for the CNN-LSTM model
 */
export const DEFAULT_CNNLSTM_PARAMS: CNNLSTMModelParams = {
  timeSteps: 10,
  features: ['close', 'open', 'high', 'low', 'volume'],
  epochs: 20,
  batchSize: 32,
  learningRate: 0.001,
  cnnFilters: 64,
  cnnKernelSize: 3,
  lstmUnits: [100, 50],
  dropoutRate: 0.2,
  denseUnits: [25, 10],
};

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

export class CNNLSTMModel {
  private model: tf.LayersModel | null = null;
  private params: CNNLSTMModelParams;
  private inputScaler: MinMaxScaler;
  private outputScaler: MinMaxScaler;
  private isTraining: boolean = false;
  private featureIndices: { [key: string]: number } = {};

  /**
   * Create a new CNN-LSTM model with given parameters
   */
  constructor(params: Partial<CNNLSTMModelParams> = {}) {
    this.params = { ...DEFAULT_CNNLSTM_PARAMS, ...params };
    this.inputScaler = new MinMaxScaler();
    this.outputScaler = new MinMaxScaler();
    
    // Map feature names to indices
    this.params.features.forEach((feature, index) => {
      this.featureIndices[feature] = index;
    });
  }

  /**
   * Build the CNN-LSTM model architecture
   */
  private buildModel(inputShape: [number, number]): tf.LayersModel {
    const model = tf.sequential();
    
    // Reshape input for CNN
    model.add(tf.layers.reshape({
      targetShape: [inputShape[0], inputShape[1], 1],
      inputShape: inputShape,
    }));
    
    // 1D CNN layer
    model.add(tf.layers.conv2d({
      filters: this.params.cnnFilters,
      kernelSize: [this.params.cnnKernelSize, 1],
      activation: 'relu',
      padding: 'same',
    }));
    
    // Reshape for LSTM
    model.add(tf.layers.reshape({
      targetShape: [inputShape[0], this.params.cnnFilters],
    }));
    
    // LSTM layers
    for (let i = 0; i < this.params.lstmUnits.length; i++) {
      model.add(tf.layers.lstm({
        units: this.params.lstmUnits[i],
        returnSequences: i < this.params.lstmUnits.length - 1,
        activation: 'tanh',
      }));
      model.add(tf.layers.dropout({ rate: this.params.dropoutRate }));
    }
    
    // Dense layers
    for (const units of this.params.denseUnits) {
      model.add(tf.layers.dense({ units, activation: 'relu' }));
      model.add(tf.layers.dropout({ rate: this.params.dropoutRate / 2 }));
    }
    
    // Output layer
    model.add(tf.layers.dense({ units: 1 }));
    
    // Compile the model
    model.compile({
      optimizer: tf.train.adam(this.params.learningRate),
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
      // Prepare features and scale data
      const features: number[][] = [];
      const targetValues: number[] = [];
      
      for (const point of data) {
        const featureValues: number[] = [];
        for (const feature of this.params.features) {
          featureValues.push(point[feature as keyof StockDataPoint] as number);
        }
        features.push(featureValues);
        targetValues.push(point.close);
      }
      
      // Fit scalers
      this.inputScaler.fit(features);
      this.outputScaler.fit(targetValues.map(v => [v]));
      
      // Scale the data points
      const scaledData = [...data];
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
      
      // Train the model
      const history = await this.model.fit(X, y, {
        epochs: this.params.epochs,
        batchSize: this.params.batchSize,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch + 1}/${this.params.epochs}, loss: ${logs?.loss.toFixed(6)}, val_loss: ${logs?.val_loss.toFixed(6)}`);
          },
        },
      });
      
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
      throw new Error(`Insufficient data for prediction. Need at least ${this.params.timeSteps} data points.`);
    }
    
    // Get the most recent data points
    const recentData = data.slice(-this.params.timeSteps);
    
    // Scale the input data
    const scaledData = [...recentData];
    for (let i = 0; i < scaledData.length; i++) {
      for (const feature of this.params.features) {
        const value = scaledData[i][feature as keyof StockDataPoint] as number;
        const featureIndex = this.featureIndices[feature];
        const scaled = this.inputScaler.transformValue(value, featureIndex);
        scaledData[i] = { ...scaledData[i], [feature]: scaled };
      }
    }
    
    // Predictions array
    const predictions: PredictionPoint[] = [];
    const lastDate = new Date(data[data.length - 1].date);
    
    // Create a copy of the scaled data that we'll update during prediction
    let currentData = [...scaledData];
    
    // Make predictions for the specified number of days
    for (let i = 0; i < days; i++) {
      // Prepare input for prediction
      const sequence: number[][] = [];
      for (let j = 0; j < this.params.timeSteps; j++) {
        const featureValues: number[] = [];
        for (const feature of this.params.features) {
          featureValues.push(currentData[j][feature as keyof StockDataPoint] as number);
        }
        sequence.push(featureValues);
      }
      
      // Convert to tensor
      const input = tf.tensor3d([sequence], [1, this.params.timeSteps, this.params.features.length]);
      
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
      const newDataPoint: Partial<StockDataPoint> = {};
      
      // Set all features to the last known values
      for (const feature of this.params.features) {
        if (feature === 'close') {
          newDataPoint[feature as keyof StockDataPoint] = predictedValue[0] as any;
        } else {
          // For other features, just use the last known value
          // This is a simplification; in a real app, you might want to predict these too
          newDataPoint[feature as keyof StockDataPoint] = 
            currentData[currentData.length - 1][feature as keyof StockDataPoint];
        }
      }
      
      // Add date to the new data point
      newDataPoint.date = predictionDate.toISOString();
      
      // Remove first element and add new prediction
      currentData.shift();
      currentData.push(newDataPoint as StockDataPoint);
      
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
}

export default CNNLSTMModel; 