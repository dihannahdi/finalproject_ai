/**
 * Configuration file for model parameters
 * This file contains fast training settings for all models
 */

// Common parameters for fast training across all models
export const FAST_TRAINING_CONFIG = {
  // Reduce timesteps (window size) for faster processing
  timeSteps: 5,
  
  // Reduce epochs for faster training
  epochs: 5,
  
  // Smaller batch size for faster iterations
  batchSize: 16,
  
  // Simplified feature set
  features: ['close', 'volume'],
};

// LSTM model fast configuration
export const FAST_LSTM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenLayers: [8, 4], // Smaller network
  learningRate: 0.002, // Slightly higher learning rate for faster convergence
  dropoutRate: 0.1,
};

// CNN-LSTM model fast configuration
export const FAST_CNNLSTM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  cnnFilters: 32, // Fewer filters
  cnnKernelSize: 2, // Smaller kernel
  lstmUnits: [32, 16], // Smaller LSTM layers
  dropoutRate: 0.1,
  denseUnits: [8, 4], // Smaller dense layers
};

// Transformer model fast configuration
export const FAST_TRANSFORMER_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  dModel: 16, // Smaller embedding dimension
  numHeads: 2, // Fewer attention heads
  numEncoderLayers: 1, // Single encoder layer
  ffnDim: 32, // Smaller feed-forward network
  dropoutRate: 0.1,
};

// TDDM model fast configuration
export const FAST_TDDM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenUnits: [32, 16], // Smaller hidden units
  dropoutRate: 0.1,
  useAttention: false, // Disable attention for speed
  useTDDMLayers: false, // Disable specialized layers for speed
  temporalFeatureExtraction: false, // Disable feature extraction for speed
};

// XGBoost model fast configuration
export const FAST_XGBOOST_CONFIG = {
  numTrees: 8,             // Fewer trees for faster training
  maxDepth: 2,             // Shallower trees
  learningRate: 0.1,       // Standard learning rate
  featureSubsamplingRatio: 0.8  // Standard feature subsampling ratio
};

// GAN model fast configuration (using LSTM as fallback)
export const FAST_GAN_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenLayers: [16, 8],   // Smaller network for faster training
  learningRate: 0.001,     // Learning rate for optimizer
  dropoutRate: 0.1,        // Dropout rate for regularization
};

// Ensemble model fast configuration
export const FAST_ENSEMBLE_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  // Use multiple models for stacking ensemble
  subModels: {
    lstm: true,
    cnnlstm: true,
    transformer: true
  },
  ensembleMethod: 'weighted', // Weighted ensemble method for better results
  weights: [0.4, 0.3, 0.3], // Weights for LSTM, CNN-LSTM, and Transformer
  epochs: 10, // Slightly more epochs for better ensemble learning
};

// Function to apply fast training config to all models based on their type
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