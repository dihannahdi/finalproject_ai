/**
 * Configuration file for model parameters
 * This file contains configurations optimized for 2-3 years of historical data
 * 
 * HOW TO TUNE ALGORITHMS:
 * 1. Modify the model-specific configurations below (FAST_LSTM_CONFIG, FAST_XGBOOST_CONFIG, etc.)
 * 2. Key parameters to tune:
 *    - timeSteps: Controls how many past data points are used for prediction (higher = more context)
 *    - epochs: Number of training iterations (higher = more learning, but risk of overfitting)
 *    - learningRate: Controls how quickly models adapt to the data (lower = more stable but slower learning)
 *    - Layer sizes (hiddenLayers, lstmUnits, etc.): Controls model complexity (larger = more capacity but needs more data)
 * 3. For specific algorithms:
 *    - LSTM/CNN-LSTM: Adjust hiddenLayers, dropoutRate
 *    - XGBoost: Adjust numTrees, maxDepth, learningRate
 *    - Transformer: Adjust numHeads, numEncoderLayers
 *    - Ensemble: Adjust weights of component models
 * 4. After changing parameters, restart the application to apply changes
 */

// Common parameters for all models with longer historical data
export const FAST_TRAINING_CONFIG = {
  // Reduce timesteps for faster processing
  timeSteps: 10,
  
  // Reduce epochs for much faster training
  epochs: 5,
  
  // Increase batch size for faster training
  batchSize: 64,
  
  // Reduce features for faster processing
  features: ['close', 'volume'],
};

// LSTM model configuration optimized for longer historical data
export const FAST_LSTM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenLayers: [16, 8], // Smaller network for faster processing
  learningRate: 0.01, // Higher learning rate for faster convergence
  dropoutRate: 0.1, // Reduced dropout for faster training
};

// CNN-LSTM model configuration optimized for longer historical data
export const FAST_CNNLSTM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  cnnFilters: 16, // Reduced from 32 for faster processing
  cnnKernelSize: 2, // Smaller kernel size for faster computation
  lstmUnits: [16, 8], // Smaller LSTM units for faster processing
  dropoutRate: 0.1,
  denseUnits: [8, 4], // Smaller dense layers for faster processing
};

// Transformer model configuration optimized for longer historical data
export const FAST_TRANSFORMER_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  dModel: 16, // Reduced embedding dimension for faster processing
  numHeads: 2, // Fewer attention heads for faster computation
  numEncoderLayers: 1, // Single encoder layer for faster processing
  ffnDim: 32, // Smaller feed-forward network
  dropoutRate: 0.1,
};

// TDDM model configuration optimized for longer historical data
export const FAST_TDDM_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenUnits: [32, 16], // Smaller hidden units for faster processing
  dropoutRate: 0.1,
  useAttention: false, // Disable attention for faster processing
  useTDDMLayers: true,
  temporalFeatureExtraction: false, // Disable feature extraction for faster processing
};

// XGBoost model configuration optimized for longer historical data
export const FAST_XGBOOST_CONFIG = {
  numTrees: 20,              // Fewer trees for faster processing
  maxDepth: 2,               // Reduced depth for faster processing
  learningRate: 0.1,         // Higher learning rate for faster convergence
  featureSubsamplingRatio: 0.6  // Reduced feature sampling for faster processing
};

// RandomForest model configuration optimized for longer historical data
export const FAST_RANDOMFOREST_CONFIG = {
  numTrees: 20,               // Fewer trees for faster processing
  maxDepth: 3,                // Reduced depth for faster processing
  featureSamplingRatio: 0.6,   // Reduced feature sampling for faster processing
  dataSamplingRatio: 0.7       // Reduced data sampling for faster processing
};

// GradientBoost model configuration optimized for longer historical data
export const FAST_GRADIENTBOOST_CONFIG = {
  numTrees: 20,            // Fewer trees for faster processing
  maxDepth: 2,             // Reduced depth for faster processing
  learningRate: 0.2,       // Higher learning rate for faster convergence
  subsampleRatio: 0.7      // Reduced subsample ratio for faster processing
};

// GAN model configuration optimized for longer historical data
export const FAST_GAN_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  hiddenLayers: [16, 8],   // Smaller network for faster processing
  learningRate: 0.01,      // Higher learning rate for faster convergence
  dropoutRate: 0.1,        // Reduced dropout for faster processing
};

// Ensemble model configuration optimized for longer historical data
export const FAST_ENSEMBLE_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  // Use fewer models for faster processing
  subModels: {
    lstm: false,
    cnnlstm: false,
    transformer: false,
    xgboost: true,
    randomforest: true,
    gradientboost: true
  },
  ensembleMethod: 'weighted', // Weighted ensemble method for better results
  weights: [0.4, 0.3, 0.3], // Weights for XGBoost, RandomForest, and GradientBoost
  epochs: 5, // Fewer epochs for faster processing
};

// Tree Ensemble model configuration (combining XGBoost, RandomForest, and GradientBoost)
export const FAST_TREES_ENSEMBLE_CONFIG = {
  ...FAST_TRAINING_CONFIG,
  // Only use tree-based models for this ensemble
  subModels: {
    lstm: false,
    cnnlstm: false,
    transformer: false,
    xgboost: true,
    randomforest: true,
    gradientboost: true
  },
  ensembleMethod: 'weighted', // Weighted ensemble method for better results
  weights: [0.4, 0.3, 0.3], // Weights for XGBoost, RandomForest, and GradientBoost
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
    case 'randomforest':
      return FAST_RANDOMFOREST_CONFIG;
    case 'gradientboost':
      return FAST_GRADIENTBOOST_CONFIG;
    case 'treesensemble':
      return FAST_TREES_ENSEMBLE_CONFIG;
    case 'gan':
      return FAST_GAN_CONFIG;
    default:
      return FAST_TRAINING_CONFIG;
  }
}

/**
 * Helper function to generate parameter combinations for hyperparameter tuning
 * This function can be used to create multiple model configurations for testing
 * 
 * @param modelType - The type of model to generate configurations for
 * @param paramToTune - The parameter to tune
 * @param values - Array of values to test for the parameter
 * @returns Array of configuration objects with different parameter values
 */
export function generateTuningConfigurations(
  modelType: string,
  paramToTune: string,
  values: any[]
): any[] {
  // Get the base configuration for this model type
  const baseConfig = getFastModelConfig(modelType);
  
  // Generate a configuration for each value
  return values.map(value => {
    // Create a deep copy of the base config
    const config = JSON.parse(JSON.stringify(baseConfig));
    
    // Handle nested parameters (e.g., 'hiddenLayers[0]')
    if (paramToTune.includes('[')) {
      const [param, indexStr] = paramToTune.split('[');
      const index = parseInt(indexStr.replace(']', ''));
      if (Array.isArray(config[param])) {
        config[param][index] = value;
      }
    } else {
      // Set the parameter to the current value
      config[paramToTune] = value;
    }
    
    return {
      modelType,
      paramValue: value,
      config
    };
  });
}

/**
 * Example usage of the tuning function:
 * 
 * // Generate configurations for tuning LSTM learning rate
 * const lstmConfigs = generateTuningConfigurations('lstm', 'learningRate', [0.0001, 0.001, 0.01]);
 * 
 * // Generate configurations for tuning XGBoost number of trees
 * const xgboostConfigs = generateTuningConfigurations('xgboost', 'numTrees', [10, 50, 100]);
 * 
 * // Generate configurations for tuning Transformer number of heads
 * const transformerConfigs = generateTuningConfigurations('transformer', 'numHeads', [2, 4, 8]);
 */ 