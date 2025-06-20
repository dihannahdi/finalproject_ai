'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  StockDataPoint, 
  PredictionPoint,
  safeGetStockData,
  safePredictStockPrice
} from '@/app/lib/api';
import StockChart from '@/app/components/StockChart';
import AlgorithmSelector from '@/app/components/AlgorithmSelector';
import PredictionResults from '@/app/components/PredictionResults';
import StockSelector from '@/app/components/StockSelector';
import DateRangePicker from '@/app/components/DateRangePicker';
import TechnicalIndicators from '@/app/components/TechnicalIndicators';
import ApiStatus from '@/app/components/ApiStatus';
import AlgorithmStatus from '@/app/components/AlgorithmStatus';
import { calculateAllAlgorithmPerformance } from '@/app/lib/algorithmPerformance';

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
  { id: 'cnnlstm', name: 'CNN-LSTM', description: 'Hybrid CNN and LSTM model' },
  { id: 'gan', name: 'GAN', description: 'Generative Adversarial Network' },
  { id: 'xgboost', name: 'XGBoost', description: 'Gradient boosting framework' },
  { id: 'randomforest', name: 'RandomForest', description: 'Random forest ensemble classifier' },
  { id: 'gradientboost', name: 'GradientBoost', description: 'Gradient boosting decision tree model' },
  { id: 'ensemble', name: 'Ensemble', description: 'Ensemble of XGBoost, RandomForest, and GradientBoost' },
];

// Mock technical indicators
const TECHNICAL_INDICATORS = [
  { id: 'sma', name: 'Simple Moving Average' },
  { id: 'ema', name: 'Exponential Moving Average' },
  { id: 'rsi', name: 'Relative Strength Index' },
  { id: 'macd', name: 'MACD' },
  { id: 'bollinger', name: 'Bollinger Bands' },
];

interface AlgorithmPerformance {
  [key: string]: string;
}

// Define proper PageProps interface
interface PageProps {
  params: { initialAlgorithm?: string };
  searchParams: { [key: string]: string | string[] | undefined };
}

export default function Dashboard({ params, searchParams }: PageProps) {
  const initialAlgorithm = params?.initialAlgorithm;
  const [isClient, setIsClient] = useState(false);
  const [stockData, setStockData] = useState<StockDataPoint[] | null>(null);
  const [predictions, setPredictions] = useState<PredictionPoint[] | null>(null);
  const [actualAlgorithm, setActualAlgorithm] = useState<string | null>(null);
  const [confidence, setConfidence] = useState('');
  const [mae, setMae] = useState('');
  const [directionalAccuracy, setDirectionalAccuracy] = useState('');
  const [algorithmPerformance, setAlgorithmPerformance] = useState<AlgorithmPerformance>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedStock, setSelectedStock] = useState(STOCK_SYMBOLS[0]);
  
  // Find the algorithm from the URL or use the default
  const getInitialAlgorithm = () => {
    if (initialAlgorithm) {
      const matchingAlgorithm = ALGORITHM_OPTIONS.find(algo => algo.id === initialAlgorithm);
      if (matchingAlgorithm) {
        return matchingAlgorithm;
      }
    }
    // Default to the first algorithm if no match is found
    return ALGORITHM_OPTIONS[0];
  };
  
  // Set the initially selected algorithm
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(getInitialAlgorithm());
  
  // Calculate default date range (last 3 years from today)
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1); // Use yesterday as the latest available date
  const threeYearsAgo = new Date(yesterday);
  threeYearsAgo.setFullYear(yesterday.getFullYear() - 3);
  
  // Format dates to YYYY-MM-DD
  const formatDate = (date: Date) => {
    return date.toISOString().split('T')[0];
  };
  
  const [dateRange, setDateRange] = useState({
    start: formatDate(threeYearsAgo),
    end: formatDate(yesterday)
  });

  // Update date range when component mounts to ensure we're using current dates
  useEffect(() => {
    const currentDate = new Date();
    const yesterday = new Date(currentDate);
    yesterday.setDate(currentDate.getDate() - 1); // Use yesterday as the latest available date
    const threeYearsAgo = new Date(yesterday);
    threeYearsAgo.setFullYear(yesterday.getFullYear() - 3);
    
    setDateRange({
      start: formatDate(threeYearsAgo),
      end: formatDate(yesterday)
    });
  }, []);
  
  const [selectedIndicators, setSelectedIndicators] = useState([TECHNICAL_INDICATORS[0].id]);
  // Add metrics state
  const [metrics, setMetrics] = useState({
    loss: '0.00',
    maePercent: '0.00',
    confidence: '0.00',
    algorithmPerformance: {} as AlgorithmPerformance
  });
  
  // Initialize metrics after component mounts (client-side only)
  useEffect(() => {
    // Generate initial metrics for algorithm performance
    setMetrics({
      loss: '0.00',
      maePercent: '0.00',
      confidence: (Math.random() * 15 + 80).toFixed(2),
      algorithmPerformance: ALGORITHM_OPTIONS.reduce<AlgorithmPerformance>((acc, algorithm) => {
        acc[algorithm.id] = ((Math.random() * 20) + (algorithm.id === selectedAlgorithm.id ? 80 : 70)).toFixed(2);
        return acc;
      }, {})
    });
  }, []);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Fetch stock data when selected stock or date range changes
  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('Fetching data for:', {
          symbol: selectedStock.symbol,
          startDate: dateRange.start,
          endDate: dateRange.end,
          algorithm: selectedAlgorithm.id
        });
        
        setIsLoading(true);
        setError('');
        
        // Use safer API call implementations
        const data = await safeGetStockData(
          selectedStock.symbol, 
          dateRange.start, 
          dateRange.end
        );
        
        console.log('Received stock data:', {
          dataLength: data?.length ?? 0,
          firstDate: data?.[0]?.date ?? 'N/A',
          lastDate: data && data.length > 0 ? data[data.length - 1]?.date : 'N/A',
          firstPrice: data?.[0]?.close ?? 0,
          lastPrice: data && data.length > 0 ? data[data.length - 1]?.close : 0
        });
        
        setStockData(data);
        
        // Use safer prediction implementation with updated return type
        const result = await safePredictStockPrice(
          selectedStock.symbol,
          data,
          selectedAlgorithm.id,
          30 // 30 days prediction
        );
        
        const { predictions, actualAlgorithm: usedAlgorithm } = result;
        
        console.log('Received predictions:', {
          predictionsLength: predictions?.length,
          firstDate: predictions?.[0]?.date,
          lastDate: predictions?.[predictions.length - 1]?.date,
          firstPrice: predictions?.[0]?.price,
          lastPrice: predictions?.[predictions.length - 1]?.price,
          actualAlgorithm: usedAlgorithm
        });
        
        setPredictions(predictions);
        setActualAlgorithm(usedAlgorithm);
        setIsLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to fetch stock data. Please try again.');
        setStockData([]);  // Initialize with empty array instead of null
        setPredictions([]); // Initialize with empty array instead of null
        setActualAlgorithm(null);
        setIsLoading(false);
      }
    };

    fetchData();
  }, [selectedStock, dateRange, selectedAlgorithm]);

  // Handler for stock selection change
  const handleStockChange = (stock: typeof STOCK_SYMBOLS[0]) => {
    setSelectedStock(stock);
  };

  // Handler for algorithm selection change
  const handleAlgorithmChange = (algorithm: typeof ALGORITHM_OPTIONS[0]) => {
    console.log(`Changing algorithm from ${selectedAlgorithm.id} to ${algorithm.id}`);
    setSelectedAlgorithm(algorithm);
  };

  // Handler for date range change
  const handleDateRangeChange = (range: { start: string, end: string }) => {
    setDateRange(range);
  };

  // Handler for indicator selection change
  const handleIndicatorChange = (indicators: string[]) => {
    setSelectedIndicators(indicators);
  };

  // Calculate metrics whenever predictions or stock data changes
  useEffect(() => {
    // Skip if we don't have enough data
    if (!predictions || predictions.length === 0 || !stockData || stockData.length === 0) {
      return;
    }

    // Calculate algorithm performance metrics
    const algorithmPerformance = calculateAlgorithmPerformance(selectedAlgorithm.id, stockData);
    
    setMetrics({
      loss: '0.00',  // Keep these properties but set to default values to avoid breaking code
      maePercent: '0.00',
      confidence: algorithmPerformance[selectedAlgorithm.id] || '0.00',
      algorithmPerformance
    });
  }, [predictions, stockData, selectedAlgorithm.id]);

  // Calculate algorithm performance based on actual metrics
  const calculateAlgorithmPerformance = (algorithmId: string, data: StockDataPoint[]): AlgorithmPerformance => {
    // Use the new service to calculate performance for all algorithms
    const result = calculateAllAlgorithmPerformance(data);
    
    // Convert to the expected format
    return result as AlgorithmPerformance;
  };
  
  // Get API key for status display
  const [apiKey, setApiKey] = useState('');
  
  useEffect(() => {
    // Check if we're in the browser
    if (typeof window !== 'undefined') {
      const key = process.env.NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY || 'demo';
      setApiKey(key);
    }
  }, []);

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
              <li><Link href="/tuning" className="text-gray-700 hover:text-primary">Tuning</Link></li>
              <li><Link href="/algorithms" className="text-gray-700 hover:text-primary">Algorithms</Link></li>
              <li><Link href="/about" className="text-gray-700 hover:text-primary">About</Link></li>
            </ul>
          </nav>
        </div>
      </header>

      {/* Dashboard Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold">Stock Prediction Dashboard</h1>
            {params?.initialAlgorithm && (
              <div className="flex items-center text-sm text-gray-600 mt-1">
                <Link href="/dashboard" className="hover:text-primary">Dashboard</Link>
                <span className="mx-2">›</span>
                <span className="font-medium text-primary">{selectedAlgorithm.name}</span>
              </div>
            )}
          </div>
          <ApiStatus apiKey={apiKey} />
        </div>

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
              <p><strong>Current Price:</strong> ${stockData && stockData.length > 0 ? stockData[stockData.length - 1].close.toFixed(2) : '0.00'}</p>
              <p><strong>Change:</strong> {stockData && stockData.length > 1 ? ((stockData[stockData.length - 1].close - stockData[stockData.length - 2].close) / stockData[stockData.length - 2].close * 100).toFixed(2) : '0.00'}%</p>
              <p><strong>Volume:</strong> {stockData && stockData.length > 0 ? stockData[stockData.length - 1].volume.toLocaleString() : '0'}</p>
            </div>
          </div>

          {/* Algorithm Selector */}
          <div className="card">
            <h3 className="text-lg font-semibold mb-3">Select Algorithm</h3>
            <AlgorithmSelector 
              algorithms={ALGORITHM_OPTIONS}
              selectedAlgorithm={selectedAlgorithm}
              onAlgorithmChange={handleAlgorithmChange}
              stockData={stockData}
            />
            
            {/* Algorithm Status Indicator */}
            <div className="mt-3">
              <AlgorithmStatus 
                requestedAlgorithm={selectedAlgorithm.id}
                actualAlgorithm={actualAlgorithm}
              />
            </div>
            
            {/* Quick Links to Algorithm Routes */}
            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Quick Access Links:</h4>
              <div className="flex flex-wrap gap-2">
                {ALGORITHM_OPTIONS.map(algo => (
                  <Link 
                    key={algo.id}
                    href={`/dashboard/${algo.id}`}
                    className={`text-xs px-2 py-1 rounded ${selectedAlgorithm.id === algo.id 
                      ? 'bg-primary text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {algo.name}
                  </Link>
                ))}
              </div>
            </div>
          </div>

          {/* Date Range Selector */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Date Range</h2>
            <div className="flex gap-4">
              <div>
                <label htmlFor="startDate" className="block text-sm font-medium text-gray-700 mb-1">
                  Start Date
                </label>
                <input
                  type="date"
                  id="startDate"
                  value={dateRange.start}
                  onChange={(e) => {
                    const newStart = e.target.value;
                    const endDate = new Date(dateRange.end);
                    const startDate = new Date(newStart);
                    
                    // Ensure start date is not after end date
                    if (startDate > endDate) {
                      setDateRange({
                        start: newStart,
                        end: newStart
                      });
                    } else {
                      setDateRange(prev => ({
                        ...prev,
                        start: newStart
                      }));
                    }
                  }}
                  max={dateRange.end}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
              </div>
              <div>
                <label htmlFor="endDate" className="block text-sm font-medium text-gray-700 mb-1">
                  End Date
                </label>
                <input
                  type="date"
                  id="endDate"
                  value={dateRange.end}
                  onChange={(e) => {
                    const newEnd = e.target.value;
                    const yesterday = new Date();
                    yesterday.setDate(yesterday.getDate() - 1);
                    const yesterdayStr = formatDate(yesterday);
                    
                    // Ensure end date is not in the future and not before start date
                    if (newEnd > yesterdayStr) {
                      setDateRange(prev => ({
                        ...prev,
                        end: yesterdayStr
                      }));
                    } else if (newEnd < dateRange.start) {
                      setDateRange(prev => ({
                        ...prev,
                        end: dateRange.start
                      }));
                    } else {
                      setDateRange(prev => ({
                        ...prev,
                        end: newEnd
                      }));
                    }
                  }}
                  max={(() => {
                    const yesterday = new Date();
                    yesterday.setDate(yesterday.getDate() - 1);
                    return formatDate(yesterday);
                  })()}
                  min={dateRange.start}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
              </div>
            </div>
            <p className="mt-2 text-xs text-gray-500">
              <span className="flex items-center">
                <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Only historical data up to yesterday is available. Future dates will be automatically adjusted.
              </span>
            </p>
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

          {!isClient ? (
            <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-gray-500">Loading dashboard...</p>
            </div>
          ) : isLoading ? (
            <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-gray-500">Loading chart data...</p>
            </div>
          ) : error ? (
            <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-red-500">{error}</p>
            </div>
          ) : (
            <div className="h-96">
              <StockChart
                historicalData={stockData || []}
                predictions={predictions || []}
                symbol={selectedStock.symbol}
                algorithmName={actualAlgorithm || selectedAlgorithm.name}
                indicators={selectedIndicators}
                isLoading={isLoading}
              />
            </div>
          )}
        </div>

        {/* Analysis & Predictions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Prediction Results */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
            {predictions && predictions.length > 0 ? (
              <div>
                <div className="mb-4">
                  <h3 className="text-lg font-medium text-gray-700">Future Price Estimates</h3>
                  <p className="text-2xl font-bold text-primary">
                    ${predictions[predictions.length - 1]?.price?.toFixed(2) || '0.00'}
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
                        <td className={`text-right py-2 ${stockData && stockData.length > 0 && prediction.price > stockData[stockData.length - 1]?.close ? 'text-green-600' : 'text-red-600'}`}>
                          {stockData && stockData.length > 0 ? ((prediction.price / stockData[stockData.length - 1]?.close - 1) * 100).toFixed(2) : '0.00'}%
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
            <div className="grid grid-cols-1 gap-4 mb-6">
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-sm text-gray-600">Accuracy</p>
                <p className="text-2xl font-bold text-primary">
                  {metrics.algorithmPerformance[selectedAlgorithm.id] || "0.00"}%
                </p>
                <p className="text-xs text-gray-500">Directional Precision</p>
              </div>
            </div>
            
            <h3 className="text-lg font-medium mb-2">Algorithm Comparison</h3>
            <p className="text-sm text-gray-600 mb-4">
              Comparative accuracy of different algorithms for {selectedStock.symbol}
            </p>
            
            <div className="space-y-3">
              {ALGORITHM_OPTIONS.map((algorithm) => {
                // Get the accuracy value for this algorithm
                const accuracy = metrics.algorithmPerformance[algorithm.id] || "0.00";
                
                // Check if this is the selected algorithm to highlight it
                const isSelected = algorithm.id === selectedAlgorithm.id;
                
                // Calculate the bar color based on accuracy value
                let barColor = 'bg-gray-400';
                if (isSelected) {
                  barColor = 'bg-primary';
                } else if (parseFloat(accuracy) >= 85) {
                  barColor = 'bg-green-500';
                } else if (parseFloat(accuracy) >= 80) {
                  barColor = 'bg-blue-500';
                } else if (parseFloat(accuracy) < 78) {
                  barColor = 'bg-orange-400';
                }
                
                return (
                  <div key={algorithm.id} className="relative pt-1">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className={`text-sm font-medium ${isSelected ? 'text-primary' : 'text-gray-700'}`}>
                          {algorithm.name}
                        </span>
                      </div>
                      <div className={`text-sm ${isSelected ? 'text-primary font-medium' : 'text-gray-600'}`}>
                        {accuracy}%
                      </div>
                    </div>
                    <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200 mt-1">
                      <div 
                        style={{ 
                          width: `${accuracy}%` 
                        }}
                        className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${barColor}`}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-gray-600">
          <p>&copy; {new Date().getFullYear()} StockPred Master. All rights reserved.</p>
          <p className="text-sm mt-2">This app is for demonstration purposes only. Not financial advice.</p>
          <div className="text-xs mt-2 mb-1">
            <span>Direct algorithm access: </span>
            {ALGORITHM_OPTIONS.map((algo, i) => (
              <span key={algo.id}>
                <Link href={`/dashboard/${algo.id}`} className="text-blue-500 hover:underline">
                  {algo.id}
                </Link>
                {i < ALGORITHM_OPTIONS.length - 1 && ", "}
              </span>
            ))}
          </div>
          <p className="text-xs">Stock data powered by <a href="https://www.alphavantage.co/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Alpha Vantage</a>.</p>
        </div>
      </footer>
    </div>
  );
} 
