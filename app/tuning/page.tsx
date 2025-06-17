'use client';

import React, { useState } from 'react';
import { runTuningExperiment, EvaluationMetrics } from '@/app/lib/tuning';
import Link from 'next/link';

export default function TuningPage() {
  const [algorithmType, setAlgorithmType] = useState('xgboost');
  const [paramToTune, setParamToTune] = useState('numTrees');
  const [paramValues, setParamValues] = useState('10, 20, 50, 100');
  const [symbol, setSymbol] = useState('AAPL');
  const [startDate, setStartDate] = useState(getDefaultStartDate());
  const [endDate, setEndDate] = useState(getDefaultEndDate());
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<EvaluationMetrics[]>([]);
  const [error, setError] = useState('');

  // Available algorithms and their tunable parameters
  const algorithmParams: Record<string, string[]> = {
    'lstm': ['learningRate', 'epochs', 'batchSize', 'hiddenLayers[0]', 'dropoutRate'],
    'xgboost': ['numTrees', 'maxDepth', 'learningRate', 'featureSubsamplingRatio'],
    'transformer': ['learningRate', 'dModel', 'numHeads', 'numEncoderLayers', 'dropoutRate'],
    'cnnlstm': ['learningRate', 'epochs', 'cnnFilters', 'lstmUnits[0]', 'dropoutRate'],
    'ensemble': ['weights[0]', 'weights[1]', 'weights[2]', 'weights[3]']
  };

  // Run tuning experiment
  const runExperiment = async () => {
    try {
      setIsRunning(true);
      setError('');
      setResults([]);

      // Parse parameter values
      const values = paramValues.split(',').map(v => {
        const trimmed = v.trim();
        // Convert to number if possible
        return isNaN(Number(trimmed)) ? trimmed : Number(trimmed);
      });

      console.log(`Running tuning experiment for ${algorithmType} - ${paramToTune}:`, values);

      const experimentResults = await runTuningExperiment(
        algorithmType,
        paramToTune,
        values,
        symbol,
        startDate,
        endDate
      );

      setResults(experimentResults);
    } catch (err) {
      console.error('Error running tuning experiment:', err);
      setError('Failed to run tuning experiment. See console for details.');
    } finally {
      setIsRunning(false);
    }
  };

  // Get appropriate parameter options for the selected algorithm
  const getParamOptions = () => {
    return algorithmParams[algorithmType] || [];
  };

  // Format date for input fields
  function getDefaultStartDate(): string {
    const today = new Date();
    const threeYearsAgo = new Date(today);
    threeYearsAgo.setFullYear(today.getFullYear() - 3);
    return threeYearsAgo.toISOString().split('T')[0];
  }
  
  function getDefaultEndDate(): string {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    return yesterday.toISOString().split('T')[0];
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-6 flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold text-primary">StockPred Master</Link>
          <nav>
            <ul className="flex space-x-6">
              <li><Link href="/" className="text-gray-700 hover:text-primary">Home</Link></li>
              <li><Link href="/dashboard" className="text-gray-700 hover:text-primary">Dashboard</Link></li>
              <li><Link href="/tuning" className="text-gray-700 hover:text-primary font-semibold">Tuning</Link></li>
              <li><Link href="/about" className="text-gray-700 hover:text-primary">About</Link></li>
            </ul>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Algorithm Tuning</h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Tuning Form */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Hyperparameter Tuning Settings</h2>

            <div className="space-y-4">
              {/* Algorithm Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Algorithm
                </label>
                <select 
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                  value={algorithmType}
                  onChange={(e) => {
                    setAlgorithmType(e.target.value);
                    // Set default parameter for the selected algorithm
                    if (algorithmParams[e.target.value]?.length > 0) {
                      setParamToTune(algorithmParams[e.target.value][0]);
                    }
                  }}
                >
                  {Object.keys(algorithmParams).map(algo => (
                    <option key={algo} value={algo}>{algo}</option>
                  ))}
                </select>
              </div>

              {/* Parameter Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Parameter to Tune
                </label>
                <select 
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                  value={paramToTune}
                  onChange={(e) => setParamToTune(e.target.value)}
                >
                  {getParamOptions().map(param => (
                    <option key={param} value={param}>{param}</option>
                  ))}
                </select>
              </div>

              {/* Parameter Values */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Parameter Values (comma-separated)
                </label>
                <input
                  type="text"
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                  value={paramValues}
                  onChange={(e) => setParamValues(e.target.value)}
                  placeholder="e.g., 10, 20, 50, 100"
                />
              </div>

              {/* Stock Symbol */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Stock Symbol
                </label>
                <input
                  type="text"
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  placeholder="e.g., AAPL"
                />
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Start Date
                  </label>
                  <input
                    type="date"
                    className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    max={endDate}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    End Date
                  </label>
                  <input
                    type="date"
                    className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    min={startDate}
                    max={getDefaultEndDate()}
                  />
                </div>
              </div>

              {/* Run Button */}
              <div className="mt-6">
                <button
                  className="w-full bg-primary text-white py-2 px-4 rounded-md hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={runExperiment}
                  disabled={isRunning}
                >
                  {isRunning ? 'Running Experiment...' : 'Run Tuning Experiment'}
                </button>
              </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 p-3 rounded-md mt-4">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Results */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Tuning Results</h2>

            {isRunning ? (
              <div className="h-64 flex items-center justify-center">
                <p className="text-gray-500">Running experiment, please wait...</p>
              </div>
            ) : results.length > 0 ? (
              <div>
                <div className="mb-4">
                  <h3 className="text-lg font-medium text-gray-700">Best Configuration</h3>
                  <p className="text-xl font-bold text-primary">
                    {paramToTune} = {results[0].parameterValue}
                  </p>
                  <p className="text-sm text-gray-600">
                    Loss: {results[0].loss.toFixed(2)} | Confidence: {results[0].confidence.toFixed(2)}%
                  </p>
                </div>

                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2">Value</th>
                      <th className="text-right py-2">Loss</th>
                      <th className="text-right py-2">Confidence</th>
                      <th className="text-right py-2">Consistency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result, index) => (
                      <tr key={index} className={`border-b border-gray-100 ${index === 0 ? 'bg-blue-50' : ''}`}>
                        <td className="py-2">{result.parameterValue}</td>
                        <td className="text-right py-2">{result.loss.toFixed(2)}</td>
                        <td className="text-right py-2">{result.confidence.toFixed(2)}%</td>
                        <td className="text-right py-2">{result.consistencyScore.toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center">
                <p className="text-gray-500">No tuning results yet. Run an experiment to see results.</p>
              </div>
            )}
          </div>
        </div>

        {/* Instructions */}
        <div className="card mt-8">
          <h2 className="text-xl font-semibold mb-4">How to Tune Algorithms</h2>
          
          <div className="prose max-w-none">
            <p>
              Algorithm tuning is the process of finding the optimal values for the parameters (hyperparameters) of a machine learning model.
              This can significantly improve prediction accuracy and model performance.
            </p>
            
            <h3>Tuning Process:</h3>
            <ol>
              <li>Select the algorithm you want to tune from the dropdown menu.</li>
              <li>Choose the parameter you want to optimize.</li>
              <li>Enter multiple values for that parameter, separated by commas.</li>
              <li>Specify the stock symbol and date range for testing.</li>
              <li>Click "Run Tuning Experiment" to evaluate all parameter values.</li>
              <li>Review the results to find the best parameter value.</li>
            </ol>
            
            <h3>Key Parameters by Algorithm:</h3>
            <ul>
              <li><strong>LSTM:</strong> learningRate (0.0001-0.01), epochs (10-50), hiddenLayers (sizes)</li>
              <li><strong>XGBoost:</strong> numTrees (10-100), maxDepth (3-10), learningRate (0.01-0.3)</li>
              <li><strong>Transformer:</strong> numHeads (2-8), numEncoderLayers (1-4)</li>
              <li><strong>CNN-LSTM:</strong> cnnFilters (32-128), lstmUnits (sizes)</li>
              <li><strong>Ensemble:</strong> weights (relative importance of each model)</li>
            </ul>
            
            <p className="text-sm text-gray-600 mt-4">
              Note: Tuning experiments may take several minutes to complete, especially for complex models
              or when testing many parameter values.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-gray-600">
          <p>&copy; {new Date().getFullYear()} StockPred Master. All rights reserved.</p>
          <p className="text-sm mt-2">Algorithm tuning is for demonstration purposes only.</p>
        </div>
      </footer>
    </div>
  );
} 