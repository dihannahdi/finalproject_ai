'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { fetchStockData } from '@/app/lib/api';
import StockChart from '@/app/components/StockChart';
import AlgorithmSelector from '@/app/components/AlgorithmSelector';
import PredictionResults from '@/app/components/PredictionResults';
import StockSelector from '@/app/components/StockSelector';
import DateRangePicker from '@/app/components/DateRangePicker';
import TechnicalIndicators from '@/app/components/TechnicalIndicators';

// Mock stock symbols for demo purposes
const STOCK_SYMBOLS = [
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'MSFT', name: 'Microsoft Corporation' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.' },
  { symbol: 'AMZN', name: 'Amazon.com, Inc.' },
  { symbol: 'TSLA', name: 'Tesla, Inc.' },
  { symbol: 'FB', name: 'Meta Platforms, Inc.' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation' },
  { symbol: 'JPM', name: 'JPMorgan Chase & Co.' },
];

// Mock algorithm options
const ALGORITHM_OPTIONS = [
  { id: 'lstm', name: 'LSTM Network', description: 'Long Short-Term Memory neural network' },
  { id: 'transformer', name: 'Transformer', description: 'Transformer-based sequence model' },
  { id: 'cnn_lstm', name: 'CNN-LSTM', description: 'Hybrid CNN and LSTM model' },
  { id: 'gan', name: 'GAN', description: 'Generative Adversarial Network' },
  { id: 'xgboost', name: 'XGBoost', description: 'Gradient boosting framework' },
  { id: 'stacking', name: 'Stacking Ensemble', description: 'Stacked ensemble of multiple base models' },
];

// Mock technical indicators
const TECHNICAL_INDICATORS = [
  { id: 'sma', name: 'Simple Moving Average' },
  { id: 'ema', name: 'Exponential Moving Average' },
  { id: 'rsi', name: 'Relative Strength Index' },
  { id: 'macd', name: 'MACD' },
  { id: 'bollinger', name: 'Bollinger Bands' },
];

export default function Dashboard() {
  const [selectedStock, setSelectedStock] = useState(STOCK_SYMBOLS[0]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(ALGORITHM_OPTIONS[0]);
  const [dateRange, setDateRange] = useState({ start: '2023-01-01', end: '2023-12-31' });
  const [stockData, setStockData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [selectedIndicators, setSelectedIndicators] = useState([TECHNICAL_INDICATORS[0].id]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch stock data when selected stock or date range changes
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        setError('');
        
        // In a real app, this would fetch from an actual API
        // const data = await fetchStockData(selectedStock.symbol, dateRange.start, dateRange.end);
        
        // For demo purposes, generate mock data
        const mockData = generateMockStockData(selectedStock.symbol, dateRange.start, dateRange.end);
        
        setStockData(mockData);
        
        // Generate mock predictions
        const mockPredictions = generateMockPredictions(mockData, selectedAlgorithm.id);
        setPredictions(mockPredictions);
        
        setIsLoading(false);
      } catch (err) {
        setError('Failed to fetch stock data. Please try again.');
        setIsLoading(false);
      }
    };

    fetchData();
  }, [selectedStock, dateRange, selectedAlgorithm]);

  // Handler for stock selection change
  const handleStockChange = (stock) => {
    setSelectedStock(stock);
  };

  // Handler for algorithm selection change
  const handleAlgorithmChange = (algorithm) => {
    setSelectedAlgorithm(algorithm);
  };

  // Handler for date range change
  const handleDateRangeChange = (range) => {
    setDateRange(range);
  };

  // Handler for indicator selection change
  const handleIndicatorChange = (indicators) => {
    setSelectedIndicators(indicators);
  };

  // Generate mock stock data for demo
  const generateMockStockData = (symbol, startDate, endDate) => {
    const days = 365;
    const basePrice = symbol === 'AAPL' ? 150 : 
                     symbol === 'MSFT' ? 280 : 
                     symbol === 'GOOGL' ? 2500 : 
                     symbol === 'AMZN' ? 3300 : 
                     symbol === 'TSLA' ? 800 : 
                     symbol === 'FB' ? 330 : 
                     symbol === 'NVDA' ? 700 : 
                     symbol === 'JPM' ? 150 : 100;
    
    const data = [];
    let currentPrice = basePrice;
    
    const startTimestamp = new Date(startDate).getTime();
    const dayDuration = 24 * 60 * 60 * 1000; // 1 day in milliseconds
    
    for (let i = 0; i < days; i++) {
      const timestamp = startTimestamp + (i * dayDuration);
      const date = new Date(timestamp).toISOString().split('T')[0];
      
      // Add some random fluctuation
      const changePercent = (Math.random() - 0.48) * 3; // Slightly biased towards up
      currentPrice = Math.max(currentPrice * (1 + changePercent / 100), 0.1);
      
      // Add some patterns to make it look more realistic
      if (i % 30 === 0) {
        currentPrice = currentPrice * (1 + (Math.random() > 0.5 ? 1 : -1) * Math.random() * 0.05);
      }
      
      const volume = Math.floor(Math.random() * 10000000) + 1000000;
      
      data.push({
        date,
        timestamp,
        open: currentPrice * (1 - Math.random() * 0.01),
        high: currentPrice * (1 + Math.random() * 0.015),
        low: currentPrice * (1 - Math.random() * 0.015),
        close: currentPrice,
        volume
      });
    }
    
    return data;
  };

  // Generate mock predictions for demo
  const generateMockPredictions = (stockData, algorithmId) => {
    if (!stockData || stockData.length === 0) return null;
    
    const lastPrice = stockData[stockData.length - 1].close;
    const predictions = [];
    
    // Use different prediction patterns based on algorithm
    const volatility = algorithmId === 'lstm' ? 0.02 : 
                      algorithmId === 'transformer' ? 0.015 : 
                      algorithmId === 'cnn_lstm' ? 0.025 : 
                      algorithmId === 'gan' ? 0.03 : 
                      algorithmId === 'xgboost' ? 0.018 : 
                      algorithmId === 'stacking' ? 0.01 : 0.02;
    
    // Generate predictions for the next 30 days
    const lastDate = new Date(stockData[stockData.length - 1].date);
    let currentPrice = lastPrice;
    
    for (let i = 1; i <= 30; i++) {
      const predictionDate = new Date(lastDate);
      predictionDate.setDate(lastDate.getDate() + i);
      
      // Create some trend based on algorithm
      let trend = 0;
      if (algorithmId === 'lstm') {
        trend = 0.001 * i; // Slight uptrend
      } else if (algorithmId === 'transformer') {
        trend = 0.0015 * i; // Stronger uptrend
      } else if (algorithmId === 'gan') {
        trend = Math.sin(i / 5) * 0.01; // Oscillating pattern
      } else if (algorithmId === 'xgboost') {
        trend = 0.0005 * i; // Very slight uptrend
      } else if (algorithmId === 'stacking') {
        trend = 0.002 * i; // Stronger uptrend
      }
      
      // Add some randomness
      const randomComponent = (Math.random() - 0.5) * volatility;
      currentPrice = currentPrice * (1 + trend + randomComponent);
      
      predictions.push({
        date: predictionDate.toISOString().split('T')[0],
        price: currentPrice,
        upper: currentPrice * (1 + volatility),
        lower: currentPrice * (1 - volatility),
      });
    }
    
    return predictions;
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-6 flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold text-primary">StockPred Master</Link>
          <nav>
            <ul className="flex space-x-6">
              <li><Link href="/" className="text-gray-700 hover:text-primary">Home</Link></li>
              <li><Link href="/dashboard" className="text-gray-700 hover:text-primary font-semibold">Dashboard</Link></li>
              <li><Link href="/algorithms" className="text-gray-700 hover:text-primary">Algorithms</Link></li>
              <li><Link href="/about" className="text-gray-700 hover:text-primary">About</Link></li>
            </ul>
          </nav>
        </div>
      </header>

      {/* Dashboard Content */}
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Stock Prediction Dashboard</h1>

        {/* Control Panel */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {/* Stock Selector */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Select Stock</h2>
            <div className="mb-4">
              <select 
                className="input" 
                value={selectedStock.symbol}
                onChange={(e) => {
                  const stock = STOCK_SYMBOLS.find(s => s.symbol === e.target.value);
                  if (stock) handleStockChange(stock);
                }}
              >
                {STOCK_SYMBOLS.map(stock => (
                  <option key={stock.symbol} value={stock.symbol}>
                    {stock.symbol} - {stock.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="text-sm text-gray-600">
              <p><strong>Current Price:</strong> ${stockData ? stockData[stockData.length - 1].close.toFixed(2) : '0.00'}</p>
              <p><strong>Change:</strong> {stockData ? ((stockData[stockData.length - 1].close - stockData[stockData.length - 2].close) / stockData[stockData.length - 2].close * 100).toFixed(2) : '0.00'}%</p>
              <p><strong>Volume:</strong> {stockData ? stockData[stockData.length - 1].volume.toLocaleString() : '0'}</p>
            </div>
          </div>

          {/* Algorithm Selector */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Select Algorithm</h2>
            <div className="mb-4">
              <select 
                className="input" 
                value={selectedAlgorithm.id}
                onChange={(e) => {
                  const algorithm = ALGORITHM_OPTIONS.find(a => a.id === e.target.value);
                  if (algorithm) handleAlgorithmChange(algorithm);
                }}
              >
                {ALGORITHM_OPTIONS.map(algorithm => (
                  <option key={algorithm.id} value={algorithm.id}>
                    {algorithm.name}
                  </option>
                ))}
              </select>
            </div>
            <p className="text-sm text-gray-600">
              {selectedAlgorithm.description}
            </p>
          </div>

          {/* Date Range Selector */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Date Range</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Start Date</label>
                <input 
                  type="date" 
                  className="input" 
                  value={dateRange.start}
                  onChange={(e) => setDateRange({...dateRange, start: e.target.value})}
                />
              </div>
              <div>
                <label className="label">End Date</label>
                <input 
                  type="date" 
                  className="input" 
                  value={dateRange.end}
                  onChange={(e) => setDateRange({...dateRange, end: e.target.value})}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Chart Section */}
        <div className="card mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Stock Price & Prediction</h2>
            <div className="flex space-x-2">
              {TECHNICAL_INDICATORS.map(indicator => (
                <label key={indicator.id} className="inline-flex items-center">
                  <input 
                    type="checkbox" 
                    className="form-checkbox h-4 w-4 text-primary" 
                    checked={selectedIndicators.includes(indicator.id)} 
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedIndicators([...selectedIndicators, indicator.id]);
                      } else {
                        setSelectedIndicators(selectedIndicators.filter(id => id !== indicator.id));
                      }
                    }} 
                  />
                  <span className="ml-2 text-sm">{indicator.name}</span>
                </label>
              ))}
            </div>
          </div>

          {isLoading ? (
            <div className="h-80 flex items-center justify-center">
              <p className="text-gray-500">Loading chart data...</p>
            </div>
          ) : error ? (
            <div className="h-80 flex items-center justify-center">
              <p className="text-red-500">{error}</p>
            </div>
          ) : (
            <div className="h-96">
              {/* This would be replaced with an actual chart component */}
              <div className="w-full h-full bg-white rounded border border-gray-200 flex items-center justify-center">
                <p className="text-xl text-gray-400">Chart showing {selectedStock.symbol} historical data and {selectedAlgorithm.name} predictions</p>
              </div>
            </div>
          )}
        </div>

        {/* Analysis & Predictions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Prediction Results */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
            {predictions ? (
              <div>
                <div className="mb-4">
                  <h3 className="text-lg font-medium text-gray-700">Future Price Estimates</h3>
                  <p className="text-2xl font-bold text-primary">
                    ${predictions[predictions.length - 1].price.toFixed(2)}
                  </p>
                  <p className="text-sm text-gray-600">
                    Predicted price in 30 days (using {selectedAlgorithm.name})
                  </p>
                </div>
                
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2">Date</th>
                      <th className="text-right py-2">Price</th>
                      <th className="text-right py-2">Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictions.filter((_, index) => index % 5 === 0).map((prediction, i) => (
                      <tr key={i} className="border-b border-gray-100">
                        <td className="py-2">{prediction.date}</td>
                        <td className="text-right py-2">${prediction.price.toFixed(2)}</td>
                        <td className={`text-right py-2 ${prediction.price > stockData[stockData.length - 1].close ? 'text-green-600' : 'text-red-600'}`}>
                          {((prediction.price / stockData[stockData.length - 1].close - 1) * 100).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-gray-500">No predictions available</p>
            )}
          </div>

          {/* Performance Metrics */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Algorithm Performance</h2>
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-sm text-gray-600">Accuracy (RMSE)</p>
                <p className="text-2xl font-bold text-primary">
                  {(Math.random() * 2 + 1).toFixed(2)}%
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-sm text-gray-600">Confidence Level</p>
                <p className="text-2xl font-bold text-primary">
                  {(Math.random() * 15 + 80).toFixed(2)}%
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-sm text-gray-600">Mean Absolute Error</p>
                <p className="text-2xl font-bold text-primary">
                  ${(Math.random() * 3 + 0.5).toFixed(2)}
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-sm text-gray-600">Directional Accuracy</p>
                <p className="text-2xl font-bold text-primary">
                  {(Math.random() * 20 + 75).toFixed(2)}%
                </p>
              </div>
            </div>
            
            <h3 className="text-lg font-medium mb-2">Algorithm Comparison</h3>
            <p className="text-sm text-gray-600 mb-4">
              Relative performance of different algorithms for {selectedStock.symbol}
            </p>
            
            <div className="space-y-3">
              {ALGORITHM_OPTIONS.map((algorithm) => (
                <div key={algorithm.id} className="relative pt-1">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm font-medium text-gray-700">{algorithm.name}</span>
                    </div>
                    <div className="text-sm text-gray-600">
                      {((Math.random() * 20) + (algorithm.id === selectedAlgorithm.id ? 80 : 70)).toFixed(2)}%
                    </div>
                  </div>
                  <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200 mt-1">
                    <div 
                      style={{ 
                        width: `${((Math.random() * 20) + (algorithm.id === selectedAlgorithm.id ? 80 : 70)).toFixed(2)}%` 
                      }}
                      className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${algorithm.id === selectedAlgorithm.id ? 'bg-primary' : 'bg-gray-400'}`}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-gray-600">
          <p>&copy; {new Date().getFullYear()} StockPred Master. All rights reserved.</p>
          <p className="text-sm mt-2">This app is for demonstration purposes only. Not financial advice.</p>
        </div>
      </footer>
    </div>
  );
} 