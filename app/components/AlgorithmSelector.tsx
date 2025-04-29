'use client';

import React from 'react';

// Algorithm option interface
interface AlgorithmOption {
  id: string;
  name: string;
  description: string;
}

interface AlgorithmSelectorProps {
  algorithms: AlgorithmOption[];
  selectedAlgorithm: AlgorithmOption;
  onAlgorithmChange: (algorithm: AlgorithmOption) => void;
}

const AlgorithmSelector: React.FC<AlgorithmSelectorProps> = ({
  algorithms,
  selectedAlgorithm,
  onAlgorithmChange,
}) => {
  return (
    <div className="card">
      <h2 className="text-xl font-semibold mb-4">Select Algorithm</h2>
      <div className="mb-4">
        <select 
          className="input" 
          value={selectedAlgorithm.id}
          onChange={(e) => {
            const algorithm = algorithms.find(a => a.id === e.target.value);
            if (algorithm) onAlgorithmChange(algorithm);
          }}
        >
          {algorithms.map(algorithm => (
            <option key={algorithm.id} value={algorithm.id}>
              {algorithm.name}
            </option>
          ))}
        </select>
      </div>
      <p className="text-sm text-gray-600">
        {selectedAlgorithm.description}
      </p>
      
      {/* Algorithm Features */}
      <div className="mt-4">
        <h3 className="text-md font-medium mb-2">Algorithm Features</h3>
        <ul className="text-sm text-gray-600 space-y-1 list-disc pl-5">
          {selectedAlgorithm.id === 'lstm' && (
            <>
              <li>Effective for sequential data patterns</li>
              <li>Captures long-term dependencies</li>
              <li>Reduces vanishing gradient problem</li>
              <li>Suitable for time-series forecasting</li>
            </>
          )}
          {selectedAlgorithm.id === 'transformer' && (
            <>
              <li>Uses self-attention mechanisms</li>
              <li>Captures complex patterns at different time scales</li>
              <li>Processes entire sequences simultaneously</li>
              <li>Highly effective for long-range dependencies</li>
            </>
          )}
          {selectedAlgorithm.id === 'cnn_lstm' && (
            <>
              <li>Combines CNN feature extraction with LSTM temporal modeling</li>
              <li>Captures both spatial and temporal patterns</li>
              <li>Reduced parameter count compared to pure LSTM</li>
              <li>Effective for complex multivariate time series</li>
            </>
          )}
          {selectedAlgorithm.id === 'gan' && (
            <>
              <li>Uses adversarial learning for market simulation</li>
              <li>Models complex market behaviors</li>
              <li>Generates realistic price scenarios</li>
              <li>Captures market regime shifts</li>
            </>
          )}
          {selectedAlgorithm.id === 'xgboost' && (
            <>
              <li>Gradient boosting algorithm with regularization</li>
              <li>Handles non-linear relationships well</li>
              <li>Works with mixed feature types</li>
              <li>Fast training and inference</li>
            </>
          )}
          {selectedAlgorithm.id === 'stacking' && (
            <>
              <li>Combines multiple base models</li>
              <li>Leverages strengths of various algorithms</li>
              <li>Reduces model variance</li>
              <li>Higher stability in different market conditions</li>
            </>
          )}
        </ul>
      </div>
      
      {/* Performance Comparison */}
      <div className="mt-4">
        <h3 className="text-md font-medium mb-2">Performance Characteristics</h3>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex items-center">
            <span className="font-medium mr-2">Accuracy:</span>
            <div className="w-24 bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-primary h-2.5 rounded-full" 
                style={{ 
                  width: `${getPerformanceMetric(selectedAlgorithm.id, 'accuracy')}%` 
                }}
              ></div>
            </div>
          </div>
          <div className="flex items-center">
            <span className="font-medium mr-2">Speed:</span>
            <div className="w-24 bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-secondary h-2.5 rounded-full" 
                style={{ 
                  width: `${getPerformanceMetric(selectedAlgorithm.id, 'speed')}%` 
                }}
              ></div>
            </div>
          </div>
          <div className="flex items-center">
            <span className="font-medium mr-2">Complexity:</span>
            <div className="w-24 bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-accent h-2.5 rounded-full" 
                style={{ 
                  width: `${getPerformanceMetric(selectedAlgorithm.id, 'complexity')}%` 
                }}
              ></div>
            </div>
          </div>
          <div className="flex items-center">
            <span className="font-medium mr-2">Adaptability:</span>
            <div className="w-24 bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-warning h-2.5 rounded-full" 
                style={{ 
                  width: `${getPerformanceMetric(selectedAlgorithm.id, 'adaptability')}%` 
                }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper function to get performance metrics based on algorithm
function getPerformanceMetric(algorithmId: string, metric: string): number {
  const metrics = {
    lstm: {
      accuracy: 85,
      speed: 60,
      complexity: 80,
      adaptability: 90
    },
    transformer: {
      accuracy: 90,
      speed: 75,
      complexity: 95,
      adaptability: 85
    },
    cnn_lstm: {
      accuracy: 87,
      speed: 65,
      complexity: 85,
      adaptability: 83
    },
    gan: {
      accuracy: 82,
      speed: 50,
      complexity: 95,
      adaptability: 88
    },
    xgboost: {
      accuracy: 80,
      speed: 95,
      complexity: 70,
      adaptability: 75
    },
    stacking: {
      accuracy: 92,
      speed: 60,
      complexity: 90,
      adaptability: 85
    }
  };
  
  return metrics[algorithmId]?.[metric] || 50;
}

export default AlgorithmSelector; 