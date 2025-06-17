/**
 * Ensemble Learning model for stock market prediction
 * Based on the research paper: Comparative Study on Stock Market Prediction using Generic CNN-LSTM and Ensemble Learning
 * Implemented using TensorFlow.js for browser-based prediction
 */

import * as tf from '@tensorflow/tfjs';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import { LSTMModel } from './LSTMModel';
import { CNNLSTMModel } from './CNNLSTMModel';
import { TransformerModel } from './TransformerModel';
import { XGBoostModel } from './XGBoostModel';
import { RandomForestModel } from './RandomForestModel';
import { GradientBoostModel } from './GradientBoostModel';
import { PredictionModel } from './index';

export interface EnsembleModelParams {
  timeSteps: number;                // Number of time steps to consider for prediction
  features: string[];               // Features to use for prediction
  subModels: {                      // The models to include in the ensemble
    lstm?: boolean;                 // Include LSTM model
    cnnlstm?: boolean;              // Include CNN-LSTM model
    transformer?: boolean;          // Include Transformer model
    xgboost?: boolean;              // Include XGBoost model
    randomforest?: boolean;         // Include Random Forest model
    gradientboost?: boolean;        // Include Gradient Boosting model
  };
  ensembleMethod: 'average' | 'weighted' | 'stacked'; // Method for combining predictions
  weights?: number[];               // Weights for each model (for weighted ensemble)
}

/**
 * Default parameters for the Ensemble model
 */
export const DEFAULT_ENSEMBLE_PARAMS: EnsembleModelParams = {
  timeSteps: 10,
  features: ['close', 'open', 'high', 'low', 'volume'],
  subModels: {
    lstm: true,
    cnnlstm: true,
    transformer: true,
    xgboost: false,
    randomforest: false,
    gradientboost: false
  },
  ensembleMethod: 'weighted',
  weights: [0.3, 0.4, 0.3] // Weights for LSTM, CNN-LSTM, and Transformer, respectively
};

/**
 * Ensemble model that combines predictions from multiple models
 */
export class EnsembleModel implements PredictionModel {
  private models: PredictionModel[] = [];
  private params: EnsembleModelParams;
  private isTraining: boolean = false;
  private modelWeights: number[] = [];

  /**
   * Create a new Ensemble model with given parameters
   */
  constructor(params: Partial<EnsembleModelParams> = {}) {
    this.params = { ...DEFAULT_ENSEMBLE_PARAMS, ...params };
    console.log('🔍 EnsembleModel created with params:', JSON.stringify(this.params));
    this.setupModels();
  }

  /**
   * Initialize the component models based on the configuration
   */
  private setupModels() {
    const { subModels, features, timeSteps } = this.params;
    
    console.log('🔍 Setting up Ensemble model with sub-models:', JSON.stringify(subModels));
    
    // Initialize the models with shared parameters
    if (subModels.lstm) {
      console.log('🔍 Adding LSTM model to ensemble');
      const lstmModel = new LSTMModel({
        timeSteps,
        features,
        hiddenLayers: [100, 50],
        dropoutRate: 0.2
      });
      this.models.push(lstmModel);
    }
    
    if (subModels.cnnlstm) {
      console.log('🔍 Adding CNN-LSTM model to ensemble');
      const cnnLstmModel = new CNNLSTMModel({
        timeSteps,
        features,
        cnnFilters: 64,
        cnnKernelSize: 3,
        lstmUnits: [100, 50],
        dropoutRate: 0.2
      });
      this.models.push(cnnLstmModel);
    }
    
    if (subModels.transformer) {
      console.log('🔍 Adding Transformer model to ensemble');
      const transformerModel = new TransformerModel({
        timeSteps,
        features,
        dModel: 64,
        numHeads: 4,
        ffnDim: 128,
        numEncoderLayers: 2,
        dropoutRate: 0.1
      });
      this.models.push(transformerModel);
    }
    
    if (subModels.xgboost) {
      console.log('🔍 Adding XGBoost model to ensemble');
      const xgboostModel = new XGBoostModel({
        numTrees: 100,
        maxDepth: 3,
        learningRate: 0.1,
        featureSubsamplingRatio: 0.8
      });
      this.models.push(xgboostModel);
    }
    
    if (subModels.randomforest) {
      console.log('🔍 Adding Random Forest model to ensemble');
      const randomForestModel = new RandomForestModel({
        numTrees: 100,
        maxDepth: 5,
        featureSamplingRatio: 0.7,
        dataSamplingRatio: 0.8
      });
      this.models.push(randomForestModel);
    }
    
    if (subModels.gradientboost) {
      console.log('🔍 Adding Gradient Boosting model to ensemble');
      const gradientBoostModel = new GradientBoostModel({
        numTrees: 100,
        maxDepth: 4,
        learningRate: 0.1,
        subsampleRatio: 0.8
      });
      this.models.push(gradientBoostModel);
    }
    
    // Validate and adjust weights if necessary
    this.setupModelWeights();
    
    console.log(`🔍 Ensemble model setup complete with ${this.models.length} sub-models`);
    if (this.models.length === 0) {
      console.warn('⚠️ No models were added to the ensemble. Check subModels configuration.');
    }
  }

  /**
   * Ensure that weights are properly configured for ensemble methods
   */
  private setupModelWeights() {
    const { weights, ensembleMethod } = this.params;
    const numModels = this.models.length;
    
    if (numModels === 0) {
      throw new Error('No models selected for ensemble. Enable at least one sub-model.');
    }
    
    // Handle weights based on ensemble method
    if (ensembleMethod === 'weighted') {
      if (weights && weights.length === numModels) {
        // Use provided weights, but normalize them to sum to 1
        const sum = weights.reduce((a, b) => a + b, 0);
        this.modelWeights = weights.map(w => w / sum);
      } else {
        // Equal weights if not specified or incorrect length
        const weight = 1 / numModels;
        this.modelWeights = Array(numModels).fill(weight);
      }
    } else if (ensembleMethod === 'average') {
      // Equal weights for averaging
      const weight = 1 / numModels;
      this.modelWeights = Array(numModels).fill(weight);
    } else if (ensembleMethod === 'stacked') {
      // Weights will be determined during training for stacked ensemble
      // Initialize with equal weights
      const weight = 1 / numModels;
      this.modelWeights = Array(numModels).fill(weight);
    }
  }

  /**
   * Train all component models
   */
  async train(data: StockDataPoint[]): Promise<any> {
    if (this.models.length === 0) {
      throw new Error('No models available for training.');
    }
    
    this.isTraining = true;
    
    try {
      // Train each model sequentially
      const trainingResults = [];
      for (let i = 0; i < this.models.length; i++) {
        console.log(`Training model ${i + 1}/${this.models.length}...`);
        const result = await this.models[i].train(data);
        trainingResults.push(result);
      }
      
      // For stacked ensemble, train a meta-model using validation set predictions
      if (this.params.ensembleMethod === 'stacked') {
        await this.trainStackedEnsemble(data);
      }
      
      return trainingResults;
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * Train a meta-model for stacked ensemble
   */
  private async trainStackedEnsemble(data: StockDataPoint[]): Promise<void> {
    // Hold out a validation set (20% of data)
    const splitIndex = Math.floor(data.length * 0.8);
    const trainData = data.slice(0, splitIndex);
    const validationData = data.slice(splitIndex);
    
    if (validationData.length < this.params.timeSteps) {
      console.warn('Validation set too small for stacked ensemble training. Using weighted ensemble instead.');
      return;
    }
    
    // Get predictions from each model on validation data
    const modelPredictions: number[][] = [];
    for (const model of this.models) {
      const predictions = await model.predict(trainData, validationData.length);
      modelPredictions.push(predictions.map(p => p.price));
    }
    
    // Extract actual values from validation data
    const actualValues = validationData.map(d => d.close);
    
    // Create a simple regression model to learn optimal weights
    const X = tf.tensor2d(modelPredictions, [modelPredictions[0].length, this.models.length]);
    const y = tf.tensor2d(actualValues, [actualValues.length, 1]);
    
    // Create and train the meta-model
    const metaModel = tf.sequential();
    metaModel.add(tf.layers.dense({
      units: 1,
      inputShape: [this.models.length],
      kernelInitializer: 'ones',
      useBias: false,
      kernelRegularizer: tf.regularizers.l1({ l1: 0.01 }),
    }));
    
    metaModel.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'meanSquaredError',
    });
    
    // Train the meta-model
    await metaModel.fit(X, y, {
      epochs: 100,
      batchSize: 32,
      verbose: 0,
    });
    
    // Extract the learned weights
    const weights = await metaModel.layers[0].getWeights()[0].data();
    
    // Ensure weights are positive and normalize them
    const positiveWeights = Array.from(weights).map(w => Math.max(0, w));
    const sum = positiveWeights.reduce((a, b) => a + b, 0);
    
    // Update model weights if sum is not zero
    if (sum > 0) {
      this.modelWeights = positiveWeights.map(w => w / sum);
      console.log('Learned ensemble weights:', this.modelWeights);
    }
    
    // Clean up tensors
    X.dispose();
    y.dispose();
    metaModel.dispose();
  }

  /**
   * Combine predictions from all models
   */
  async predict(data: StockDataPoint[], days: number = 30): Promise<PredictionPoint[]> {
    console.log('🔍 EnsembleModel.predict called with:', {
      dataLength: data.length,
      days,
      numModels: this.models.length,
      modelWeights: this.modelWeights
    });
    
    if (this.models.length === 0) {
      throw new Error('No models available for prediction.');
    }
    
    // Get individual model predictions
    const modelPredictions: PredictionPoint[][] = [];
    for (let i = 0; i < this.models.length; i++) {
      console.log(`🔍 Getting predictions from model ${i+1}/${this.models.length}`);
      const predictions = await this.models[i].predict(data, days);
      modelPredictions.push(predictions);
    }
    
    // Combine predictions using the specified ensemble method
    const combinedPredictions: PredictionPoint[] = [];
    
    for (let day = 0; day < days; day++) {
      const predictionValues: number[] = modelPredictions.map(modelPred => modelPred[day].price);
      
      // Combine predictions based on the ensemble method
      let finalPrediction: number;
      
      if (this.params.ensembleMethod === 'average' || this.params.ensembleMethod === 'weighted') {
        // Weighted sum (average is just equal weights)
        finalPrediction = predictionValues.reduce(
          (sum, pred, i) => sum + pred * this.modelWeights[i], 
          0
        );
      } else if (this.params.ensembleMethod === 'stacked') {
        // Stacked ensemble uses weights learned from data
        finalPrediction = predictionValues.reduce(
          (sum, pred, i) => sum + pred * this.modelWeights[i], 
          0
        );
      } else {
        // Default to average if method not recognized
        finalPrediction = predictionValues.reduce((sum, val) => sum + val, 0) / predictionValues.length;
      }
      
      // Create the prediction point using the date from the first model's prediction
      combinedPredictions.push({
        date: modelPredictions[0][day].date,
        price: finalPrediction,
        upper: finalPrediction * 1.02, // Add 2% for upper bound
        lower: finalPrediction * 0.98, // Subtract 2% for lower bound
      });
    }
    
    return combinedPredictions;
  }

  /**
   * Save all component models
   */
  async saveModel(modelName: string): Promise<any> {
    const saveResults = [];
    
    for (let i = 0; i < this.models.length; i++) {
      const result = await this.models[i].saveModel(`${modelName}_${i}`);
      saveResults.push(result);
    }
    
    // Save ensemble weights
    localStorage.setItem(`${modelName}_ensemble_weights`, JSON.stringify(this.modelWeights));
    localStorage.setItem(`${modelName}_ensemble_config`, JSON.stringify({
      method: this.params.ensembleMethod,
      modelTypes: Object.keys(this.params.subModels).filter(
        key => this.params.subModels[key as keyof typeof this.params.subModels]
      )
    }));
    
    return saveResults;
  }

  /**
   * Load all component models
   */
  async loadModel(modelName: string): Promise<void> {
    // Load ensemble configuration
    const weightsJson = localStorage.getItem(`${modelName}_ensemble_weights`);
    const configJson = localStorage.getItem(`${modelName}_ensemble_config`);
    
    if (weightsJson && configJson) {
      this.modelWeights = JSON.parse(weightsJson);
      const config = JSON.parse(configJson);
      this.params.ensembleMethod = config.method;
      
      // Recreate the models based on saved configuration if needed
      if (this.models.length === 0) {
        this.setupModels();
      }
    }
    
    // Load individual models
    for (let i = 0; i < this.models.length; i++) {
      await this.models[i].loadModel(`${modelName}_${i}`);
    }
    
    console.log('Ensemble model loaded successfully');
  }

  /**
   * Get the weights assigned to each model
   */
  getModelWeights(): number[] {
    return [...this.modelWeights];
  }

  /**
   * Get the component models
   */
  getModels(): PredictionModel[] {
    return [...this.models];
  }
}

export default EnsembleModel; 