/**
 * API utilities for fetching stock market data
 */

import axios from 'axios';

// API endpoints for stock data
const ALPHA_VANTAGE_API_URL = 'https://www.alphavantage.co/query';
const ALPHA_VANTAGE_API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'; // Replace with your actual API key

// Types
export interface StockDataPoint {
  date: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionPoint {
  date: string;
  price: number;
  upper: number;
  lower: number;
}

/**
 * Fetch historical stock data from Alpha Vantage API
 */
export async function fetchStockData(
  symbol: string,
  startDate: string,
  endDate: string
): Promise<StockDataPoint[]> {
  try {
    // In a real application, you would use your API key
    const response = await axios.get(ALPHA_VANTAGE_API_URL, {
      params: {
        function: 'TIME_SERIES_DAILY',
        symbol,
        apikey: ALPHA_VANTAGE_API_KEY,
        outputsize: 'full',
      },
    });

    // For demo purposes, returning a placeholder message
    console.log('API call would fetch data for:', symbol, startDate, endDate);
    
    // In a real application, parse the response
    // const result = response.data['Time Series (Daily)'];
    // const stockData: StockDataPoint[] = [];
    
    // for (const date in result) {
    //   if (date >= startDate && date <= endDate) {
    //     const data = result[date];
    //     stockData.push({
    //       date,
    //       timestamp: new Date(date).getTime(),
    //       open: parseFloat(data['1. open']),
    //       high: parseFloat(data['2. high']),
    //       low: parseFloat(data['3. low']),
    //       close: parseFloat(data['4. close']),
    //       volume: parseInt(data['5. volume']),
    //     });
    //   }
    // }
    
    // Return mock data for demo
    return generateMockStockData(symbol, startDate, endDate);
  } catch (error) {
    console.error('Error fetching stock data:', error);
    return generateMockStockData(symbol, startDate, endDate);
  }
}

/**
 * Generate mock stock data for demonstration purposes
 */
function generateMockStockData(
  symbol: string,
  startDate: string,
  endDate: string
): StockDataPoint[] {
  const days = 365;
  const basePrice = symbol === 'AAPL' ? 150 : 
                   symbol === 'MSFT' ? 280 : 
                   symbol === 'GOOGL' ? 2500 : 
                   symbol === 'AMZN' ? 3300 : 
                   symbol === 'TSLA' ? 800 : 
                   symbol === 'FB' ? 330 : 
                   symbol === 'NVDA' ? 700 : 
                   symbol === 'JPM' ? 150 : 100;
  
  const data: StockDataPoint[] = [];
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
}

/**
 * Fetch real-time stock quote data from Alpha Vantage API
 */
export async function fetchRealTimeStockData(symbol: string) {
  try {
    const response = await axios.get(ALPHA_VANTAGE_API_URL, {
      params: {
        function: 'GLOBAL_QUOTE',
        symbol,
        apikey: ALPHA_VANTAGE_API_KEY,
      },
    });

    // For demo purposes, returning a placeholder message
    console.log('API call would fetch real-time data for:', symbol);
    
    // In a real application, parse the response
    // const quote = response.data['Global Quote'];
    // return {
    //   symbol: quote['01. symbol'],
    //   price: parseFloat(quote['05. price']),
    //   change: parseFloat(quote['09. change']),
    //   changePercent: parseFloat(quote['10. change percent'].replace('%', '')),
    //   volume: parseInt(quote['06. volume']),
    // };
    
    // Return mock data for demo
    return {
      symbol,
      price: Math.random() * 1000,
      change: Math.random() * 10 - 5,
      changePercent: Math.random() * 5 - 2.5,
      volume: Math.floor(Math.random() * 10000000),
    };
  } catch (error) {
    console.error('Error fetching real-time stock data:', error);
    throw error;
  }
}

/**
 * Function to access different algorithm models for stock prediction
 */
export async function predictStockPrice(
  symbol: string,
  historicalData: StockDataPoint[],
  algorithm: string,
  predictionDays: number = 30
): Promise<PredictionPoint[]> {
  // In a real application, this would call a machine learning service
  // with the appropriate algorithm implementation
  
  console.log(`Predicting stock prices for ${symbol} using ${algorithm} algorithm`);
  
  // Return mock predictions for demo
  return generateMockPredictions(historicalData, algorithm, predictionDays);
}

/**
 * Generate mock predictions for demonstration purposes
 */
function generateMockPredictions(
  stockData: StockDataPoint[],
  algorithmId: string,
  days: number = 30
): PredictionPoint[] {
  if (!stockData || stockData.length === 0) return [];
  
  const lastPrice = stockData[stockData.length - 1].close;
  const predictions: PredictionPoint[] = [];
  
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
  
  for (let i = 1; i <= days; i++) {
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
} 