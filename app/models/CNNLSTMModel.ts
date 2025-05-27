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
   * Train the CNN-LSTM model with historical stock data
   */
  async train(data: StockDataPoint[]): Promise<tf.History> {
    try {
      if (this.isTraining) {
        console.warn('Model is already training. Please wait for the current training to complete.');
        throw new Error('Model is already training');
      }
      
      this.isTraining = true;
      console.log(`[CNNLSTMModel] Starting training with ${data.length} data points`);
      
      try {
        // Extract feature data
        const featureData: number[][] = [];
        const targetData: number[] = [];
        
        console.log(`[CNNLSTMModel] Extracting features: ${this.params.features.join(', ')}`);
        
        // Extract features from data
        for (const point of data) {
          const features: number[] = [];
          for (const feature of this.params.features) {
            const value = point[feature as keyof StockDataPoint] as number;
            if (!isFinite(value)) {
              // Handle invalid feature values
              console.warn(`[CNNLSTMModel] Invalid ${feature} value found, using default`);
              features.push(feature === 'volume' ? 1000000 : point.close);
            } else {
              features.push(value);
            }
          }
          featureData.push(features);
          targetData.push(point.close);
        }
        
        // Fit scalers
        console.log(`[CNNLSTMModel] Fitting scalers with ${featureData.length} data points`);
        this.inputScaler.fit(featureData);
        this.outputScaler.fit([[Math.min(...targetData)], [Math.max(...targetData)]]);
        
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
        console.log(`[CNNLSTMModel] Preparing training data with timeSteps=${this.params.timeSteps}`);
        const { X, y } = this.prepareTrainingData(scaledData);
        
        // Build model with the correct input shape
        console.log(`[CNNLSTMModel] Building model with shape [${this.params.timeSteps}, ${this.params.features.length}]`);
        this.model = this.buildModel([this.params.timeSteps, this.params.features.length]);
        
        // Train the model
        console.log(`[CNNLSTMModel] Starting model training with ${this.params.epochs} epochs`);
        const history = await this.model.fit(X, y, {
          epochs: this.params.epochs,
          batchSize: this.params.batchSize,
          validationSplit: 0.1,
          callbacks: {
            onEpochBegin: (epoch) => {
              console.log(`[CNNLSTMModel] Starting epoch ${epoch + 1}/${this.params.epochs}`);
            },
            onEpochEnd: (epoch, logs) => {
              console.log(`[CNNLSTMModel] Epoch ${epoch + 1}/${this.params.epochs}, loss: ${logs?.loss.toFixed(6)}, val_loss: ${logs?.val_loss.toFixed(6)}`);
            },
          },
        });
        
        console.log(`[CNNLSTMModel] Training completed successfully`);
        
        // Clean up tensors
        X.dispose();
        y.dispose();
        
        return history;
      } finally {
        this.isTraining = false;
      }
    } catch (error) {
      console.error(`[CNNLSTMModel] Error during training:`, error);
      this.isTraining = false;
      throw error;
    }
  }

  /**
   * Generate price predictions using the trained CNN-LSTM model
   */
  async predict(data: StockDataPoint[], days: number = 30): Promise<PredictionPoint[]> {
    try {
      if (!this.model) {
        console.error('[CNNLSTMModel] Model not trained yet');
        throw new Error('Model not trained yet. Call train() first.');
      }
      
      console.log(`[CNNLSTMModel] Starting prediction for ${days} days`);
      
      if (data.length < this.params.timeSteps) {
        console.error(`[CNNLSTMModel] Not enough data points for prediction`);
        throw new Error(`Not enough data points. Need at least ${this.params.timeSteps} data points.`);
      }
      
      // Create a copy of the last timeSteps data points for prediction
      const lastDataPoints = data.slice(-this.params.timeSteps);
      console.log(`[CNNLSTMModel] Using last ${lastDataPoints.length} data points for prediction`);
      
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
      
      console.log(`[CNNLSTMModel] Making predictions for ${days} days ahead`);
      
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
        const volatility = 0.02 * Math.sqrt((i + 1) / 10); // Increasing volatility over time
        
        // Create prediction point
        const predictionPoint: PredictionPoint = {
          date: predictionDate.toISOString().split('T')[0],
          price: priceValue,
          upper: priceValue * (1 + volatility),
          lower: priceValue * (1 - volatility),
          algorithmUsed: 'cnnlstm',
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
      
      console.log(`[CNNLSTMModel] Successfully generated ${predictions.length} predictions`);
      
      return predictions;
    } catch (error) {
      console.error(`[CNNLSTMModel] Error during prediction:`, error);
      throw error;
    }
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