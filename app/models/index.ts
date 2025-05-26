/**
 * Models index file to export all available prediction models
 */

import LSTMModel, { LSTMModelParams } from './LSTMModel';
import TransformerModel, { TransformerModelParams } from './TransformerModel';
import CNNLSTMModel, { CNNLSTMModelParams } from './CNNLSTMModel';
import EnsembleModel, { EnsembleModelParams } from './EnsembleModel';
import TDDMModel, { TDDMModelParams } from './TDDMModel';
import XGBoostModel, { XGBoostModelParams } from './XGBoostModel';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import { getFastModelConfig } from './config';

// Factory function to create a model instance based on algorithm type
export function createModel(algorithm: string, params: any = {}) {
  // Get the fast configuration for the specified algorithm type
  const fastConfig = getFastModelConfig(algorithm);
  
  // Merge provided params with fast config (user params take precedence)
  const mergedParams = { ...fastConfig, ...params };
  
  console.log(`Creating ${algorithm} model with fast training configuration:`, 
    `epochs=${mergedParams.epochs}, timeSteps=${mergedParams.timeSteps}`);
  
  const modelType = algorithm.toLowerCase();
  
  switch (modelType) {
    case 'lstm':
      return new LSTMModel(mergedParams);
    case 'transformer':
      return new TransformerModel(mergedParams);
    case 'cnnlstm':
      return new CNNLSTMModel(mergedParams);
    case 'ensemble':
      return new EnsembleModel(mergedParams);
    case 'tddm':
      return new TDDMModel(mergedParams);
    case 'xgboost':
      return new XGBoostModel(mergedParams);
    case 'gan':
      console.log('GAN model selected. Currently using LSTM as a fallback since GAN is not fully implemented.');
      return new LSTMModel(mergedParams);
    // Add other model types as they are implemented
    default:
      // Default to LSTM if the algorithm is not recognized
      console.warn(`Algorithm '${algorithm}' not recognized, using LSTM instead.`);
      return new LSTMModel(mergedParams);
  }
}

// Export all models
export { LSTMModel, TransformerModel, CNNLSTMModel, EnsembleModel, TDDMModel, XGBoostModel };

// Export model parameter types
export type { 
  LSTMModelParams, 
  TransformerModelParams, 
  CNNLSTMModelParams, 
  EnsembleModelParams,
  TDDMModelParams,
  XGBoostModelParams
};

// Export interface for generic model
export interface PredictionModel {
  train: (data: StockDataPoint[]) => Promise<any>;
  predict: (data: StockDataPoint[], days?: number) => Promise<PredictionPoint[]>;
  saveModel: (modelName: string) => Promise<any>;
  loadModel: (modelName: string) => Promise<void>;
} 