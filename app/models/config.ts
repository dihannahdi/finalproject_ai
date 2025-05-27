/**
 * Configuration file for model parameters
 * This file contains configurations optimized for 2-3 years of historical data
 */

// Common parameters for all models with longer historical data
export const FAST_TRAINING_CONFIG = {
  // Increase timesteps for better pattern recognition with more data
  timeSteps: 20,
  
  // Increase epochs for better training with more data
  epochs: 15,
  
  // Slightly larger batch size for efficiency with more data
  batchSize: 32,
  
  // Extended feature set for better predictions
  features: ['close', 'high', 'low', 'volume'],
};

// LSTM model configuration optimized for longer historical data
export const FAST_LSTM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenLayers: [32, 16], // Larger network for more data
  learningRate: 0.001, // Balanced learning rate
  dropoutRate: 0.2, // Increased dropout to prevent overfitting with more data
};

// CNN-LSTM model configuration optimized for longer historical data
export const FAST_CNNLSTM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  cnnFilters: 64, // More filters to capture patterns
  cnnKernelSize: 3, // Larger kernel for better feature extraction
  lstmUnits: [64, 32], // Larger LSTM layers
  dropoutRate: 0.2,
  denseUnits: [16, 8], // Larger dense layers
};

// Transformer model configuration optimized for longer historical data
export const FAST_TRANSFORMER_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  dModel: 32, // Increased embedding dimension
  numHeads: 4, // More attention heads for complex patterns
  numEncoderLayers: 2, // Multiple encoder layers
  ffnDim: 64, // Larger feed-forward network
  dropoutRate: 0.2,
};

// TDDM model configuration optimized for longer historical data
export const FAST_TDDM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenUnits: [64, 32], // Larger hidden units
  dropoutRate: 0.2,
  useAttention: true, // Enable attention for better pattern recognition with more data
  useTDDMLayers: true, // Enable specialized layers for better performance
  temporalFeatureExtraction: true, // Enable feature extraction with more data available
};

// XGBoost model configuration optimized for longer historical data
export const FAST_XGBOOST_CONFIG = {
  numTrees: 50,              // More trees for better ensemble performance and stability
  maxDepth: 3,               // Reduced depth to prevent overfitting
  learningRate: 0.05,        // Lower learning rate for more stable predictions
  featureSubsamplingRatio: 0.7  // Reduced feature sampling to prevent overfitting
};

// GAN model configuration optimized for longer historical data
export const FAST_GAN_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenLayers: [32, 16],   // Larger network for more complex patterns
  learningRate: 0.001,      // Learning rate for optimizer
  dropoutRate: 0.2,         // Increased dropout for regularization
};

// Ensemble model configuration optimized for longer historical data
export const FAST_ENSEMBLE_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  // Use multiple models for stacking ensemble
  subModels: {
    lstm: true,
    cnnlstm: true,
    transformer: true,
    xgboost: true
  },
  ensembleMethod: 'weighted', // Weighted ensemble method for better results
  weights: [0.3, 0.2, 0.2, 0.3], // Weights for LSTM, CNN-LSTM, Transformer, and XGBoost
  epochs: 20, // More epochs for better ensemble learning with more data
};

// Function to apply optimized config to all models based on their type
export function getFastModelConfig(modelType: string) {
  switch (modelType.toLowerCase()) {
    case 'lstm':
      return FAST_LSTM_CONFIG;
    case 'cnnlstm':
      return FAST_CNNLSTM_CONFIG;
    case 'transformer':
      return FAST_TRANSFORMER_CONFIG;
    case 'tddm':
      return FAST_TDDM_CONFIG;
    case 'ensemble':
      return FAST_ENSEMBLE_CONFIG;
    case 'xgboost':
      return FAST_XGBOOST_CONFIG;
    case 'gan':
      return FAST_GAN_CONFIG;
    default:
      return FAST_TRAINING_CONFIG;
  }
} 