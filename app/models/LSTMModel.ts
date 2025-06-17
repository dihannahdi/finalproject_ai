/**
 * LSTM (Long Short-Term Memory) model for stock price prediction
 * Implemented using TensorFlow.js for browser-based prediction
 */

import * as tf from '@tensorflow/tfjs';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import { PredictionModel } from './index';

export interface LSTMModelParams {
  timeSteps: number;           // Number of time steps to consider for prediction
  features: string[];          // Features to use for prediction (e.g., ['close', 'volume'])
  epochs: number;              // Number of training epochs
  batchSize: number;           // Batch size for training
  learningRate: number;        // Learning rate for optimizer
  hiddenLayers: number[];      // Array defining hidden layer sizes
  dropoutRate: number;         // Dropout rate for regularization
}

/**
 * Default parameters for the LSTM model
 */
export const DEFAULT_LSTM_PARAMS: LSTMModelParams = {
  timeSteps: 10,
  features: ['close', 'open', 'high', 'low', 'volume'],
  epochs: 20,
  batchSize: 32,
  learningRate: 0.001,
  hiddenLayers: [16, 8],
  dropoutRate: 0.2
};

export class LSTMModel implements PredictionModel {
  private model: tf.LayersModel | null = null;
  private params: LSTMModelParams;
  private inputScaler: MinMaxScaler;
  private outputScaler: MinMaxScaler;
  private isTraining: boolean = false;
  private featureIndices: { [key: string]: number } = {};

  /**
   * Create a new LSTM model with given parameters
   */
  constructor(params: Partial<LSTMModelParams> = {}) {
    this.params = { ...DEFAULT_LSTM_PARAMS, ...params };
    this.inputScaler = new MinMaxScaler();
    this.outputScaler = new MinMaxScaler();
    
    // Map feature names to indices
    this.params.features.forEach((feature, index) => {
      this.featureIndices[feature] = index;
    });
  }

  /**
   * Build the LSTM model architecture
   */
  private buildModel(inputShape: [number, number]): tf.LayersModel {
    const model = tf.sequential();
    
    // Input layer
    model.add(tf.layers.lstm({
      units: this.params.hiddenLayers[0],
      returnSequences: this.params.hiddenLayers.length > 1,
      inputShape,
      activation: 'relu',
    }));
    
    model.add(tf.layers.dropout({ rate: this.params.dropoutRate }));
    
    // Hidden layers
    for (let i = 1; i < this.params.hiddenLayers.length; i++) {
      model.add(tf.layers.lstm({
        units: this.params.hiddenLayers[i],
        returnSequences: i < this.params.hiddenLayers.length - 1,
        activation: 'relu',
      }));
      model.add(tf.layers.dropout({ rate: this.params.dropoutRate }));
    }
    
    // Output layer
    model.add(tf.layers.dense({ units: 1 }));
    
    // Compile the model
    model.compile({
      optimizer: tf.train.adam(this.params.learningRate),
      loss: 'meanSquaredError',
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
        validationSplit: 0.1,
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
    
    // Make predictions for the specified number of days
    const predictions: PredictionPoint[] = [];
    let currentSequence = scaledData;
    
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
      const priceValue = this.outputScaler.inverseTransformValue(predictionValue, 0);
      
      // Create a date for this prediction (1 day after the last date)
      const lastDate = new Date(
        i === 0 
          ? data[data.length - 1].date 
          : predictions[i - 1].date
      );
      const predictionDate = new Date(lastDate);
      predictionDate.setDate(lastDate.getDate() + 1);
      
      // Add some volatility for upper and lower bounds
      const volatility = 0.02; // 2% volatility
      
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
   * Transform a single value for a specific feature
   */
  transformValue(value: number, featureIndex: number): number {
    return (value - this.min[featureIndex]) / this.range[featureIndex];
  }
  
  /**
   * Invert the transformation for a single value
   */
  inverseTransformValue(value: number, featureIndex: number): number {
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

export default LSTMModel; 