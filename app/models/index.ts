/**
 * Models index file to export all available prediction models
 */

import { LSTMModel, LSTMModelParams } from './LSTMModel';
import { TransformerModel, TransformerModelParams } from './TransformerModel';
import { CNNLSTMModel, CNNLSTMModelParams } from './CNNLSTMModel';
import { EnsembleModel, EnsembleModelParams } from './EnsembleModel';
import { TDDMModel, TDDMModelParams } from './TDDMModel';
import { XGBoostModel, XGBoostModelParams } from './XGBoostModel';
import { RandomForestModel, RandomForestModelParams } from './RandomForestModel';
import { GradientBoostModel, GradientBoostModelParams } from './GradientBoostModel';
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
    case 'randomforest':
      return new RandomForestModel(mergedParams);
    case 'gradientboost':
      return new GradientBoostModel(mergedParams);
    case 'treesensemble':
      // Special ensemble of tree-based models
      const treesEnsembleParams: EnsembleModelParams = {
        ...mergedParams,
        subModels: {
          lstm: false,
          cnnlstm: false,
          transformer: false,
          xgboost: true,
          randomforest: true,
          gradientboost: true
        },
        ensembleMethod: 'weighted',
        weights: [0.4, 0.3, 0.3] // XGBoost, RandomForest, GradientBoost
      };
      return new EnsembleModel(treesEnsembleParams);
    case 'gan':
      console.warn('GAN model selected. Currently using LSTM as a fallback since GAN is not fully implemented.');
      // Create a special version of LSTM that identifies as GAN in its predict method
      const lstmModel = new LSTMModel(mergedParams);
      const originalPredict = lstmModel.predict;
      
      // Override the predict method to mark predictions as coming from GAN
      lstmModel.predict = async (data: StockDataPoint[], days?: number) => {
        const predictions = await originalPredict.call(lstmModel, data, days);
        // Mark each prediction as coming from GAN (LSTM fallback)
        return predictions.map(p => ({
          ...p,
          algorithmUsed: 'gan (lstm fallback)'
        }));
      };
      
      return lstmModel;
    // Add other model types as they are implemented
    default:
      // Default to LSTM if the algorithm is not recognized
      console.warn(`Algorithm '${algorithm}' not recognized, using LSTM instead.`);
      return new LSTMModel(mergedParams);
  }
}

// Export interface for generic model
export interface PredictionModel {
  train: (data: StockDataPoint[]) => Promise<any>;
  predict: (data: StockDataPoint[], days?: number) => Promise<PredictionPoint[]>;
  saveModel: (modelName: string) => Promise<any>;
  loadModel: (modelName: string) => Promise<void>;
}

// Export all models
export {
  LSTMModel,
  TransformerModel,
  CNNLSTMModel,
  EnsembleModel,
  TDDMModel,
  XGBoostModel,
  RandomForestModel,
  GradientBoostModel
};

// Export model parameter types
export type { 
  LSTMModelParams, 
  TransformerModelParams, 
  CNNLSTMModelParams, 
  EnsembleModelParams,
  TDDMModelParams,
  XGBoostModelParams,
  RandomForestModelParams,
  GradientBoostModelParams
}; 