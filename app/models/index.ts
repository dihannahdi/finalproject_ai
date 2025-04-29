/**
 * Models index file to export all available prediction models
 */

import LSTMModel, { LSTMModelParams } from './LSTMModel';
import TransformerModel, { TransformerModelParams } from './TransformerModel';

// Factory function to create a model instance based on algorithm type
export function createModel(algorithm: string, params: any = {}) {
  switch (algorithm.toLowerCase()) {
    case 'lstm':
      return new LSTMModel(params);
    case 'transformer':
      return new TransformerModel(params);
    // Add other model types as they are implemented
    default:
      // Default to LSTM if the algorithm is not recognized
      console.warn(`Algorithm '${algorithm}' not recognized, using LSTM instead.`);
      return new LSTMModel(params);
  }
}

// Export all models
export { LSTMModel, TransformerModel };

// Export model parameter types
export type { LSTMModelParams, TransformerModelParams };

// Export interface for generic model
export interface PredictionModel {
  train: (data: any[]) => Promise<any>;
  predict: (data: any[], days?: number) => Promise<any[]>;
  saveModel: (modelName: string) => Promise<any>;
  loadModel: (modelName: string) => Promise<void>;
} 