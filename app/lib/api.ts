/**
 * API utilities for fetching stock market data
 */

import axios from 'axios';
import * as tf from '@tensorflow/tfjs';
import { createModel } from '@/app/models';
import { LSTMModel } from '../models/LSTMModel';
import { 
  TransformerModel, 
  CNNLSTMModel, 
  EnsembleModel, 
  TDDMModel,
  XGBoostModel
} from '@/app/models';

// API endpoints for stock data
const ALPHA_VANTAGE_API_URL = 'https://www.alphavantage.co/query';
const ALPHA_VANTAGE_API_KEY = process.env.NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY || 'demo';

if (!ALPHA_VANTAGE_API_KEY || ALPHA_VANTAGE_API_KEY === 'demo') {
  console.warn('Using Alpha Vantage demo API key. For full functionality, please set NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY in your environment variables.');
  console.warn('With the demo key, you may experience limited data access and rate limiting.');
}

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
  algorithmUsed?: string; // Track which algorithm was actually used
}

// Helper function to check if API key is valid
function isApiKeyValid(): boolean {
  return !!ALPHA_VANTAGE_API_KEY && ALPHA_VANTAGE_API_KEY.length > 3;
}

/**
 * Fetch historical stock data from Alpha Vantage API
 * 
 * @param symbol - The name of the equity (e.g., 'IBM', 'TSCO.LON', 'SHOP.TRT')
 * @param startDate - Start date for filtering results (YYYY-MM-DD)
 * @param endDate - End date for filtering results (YYYY-MM-DD)
 * @param outputSize - Optional: 'compact' (latest 100 data points) or 'full' (20+ years of data)
 * @param dataType - Optional: 'json' or 'csv' format for the response
 */
export async function fetchStockData(
  symbol: string,
  startDate: string,
  endDate: string,
  outputSize: 'compact' | 'full' = 'compact',
  dataType: 'json' | 'csv' = 'json'
): Promise<StockDataPoint[]> {
  try {
    // Input validation
    if (!symbol || typeof symbol !== 'string' || symbol.trim() === '') {
      console.warn('Invalid symbol provided to fetchStockData');
      return generateMockStockData(symbol, startDate, endDate);
    }

    // Validate and adjust date ranges
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];
    
    // Ensure endDate is not in the future
    if (endDate > yesterdayStr) {
      console.warn(`End date ${endDate} is in the future, adjusting to yesterday (${yesterdayStr})`);
      endDate = yesterdayStr;
    }
    
    // Ensure startDate is not after endDate
    if (startDate > endDate) {
      console.warn(`Start date ${startDate} is after end date ${endDate}, adjusting to one year before end date`);
      const oneYearBefore = new Date(endDate);
      oneYearBefore.setFullYear(oneYearBefore.getFullYear() - 1);
      startDate = oneYearBefore.toISOString().split('T')[0];
    }

    console.log('Fetching stock data for:', symbol, startDate, endDate);
    
    try {
      // First, try using the TIME_SERIES_DAILY endpoint
      const response = await axios.get(ALPHA_VANTAGE_API_URL, {
        params: {
          function: 'TIME_SERIES_DAILY',
          symbol,
          outputsize: outputSize,
          datatype: dataType,
          apikey: ALPHA_VANTAGE_API_KEY
        },
        timeout: 10000 // 10 second timeout
      });
      
      // Alpha Vantage may return an error message in the response
      if (response.data && response.data['Error Message']) {
        console.warn('Alpha Vantage API error:', response.data['Error Message']);
        return generateMockStockData(symbol, startDate, endDate);
      }
      
      // Check for rate limiting
      if (response.data && response.data['Note'] && response.data['Note'].includes('API call frequency')) {
        console.warn('Alpha Vantage API rate limit reached:', response.data['Note']);
        return generateMockStockData(symbol, startDate, endDate);
      }
      
      // Check if we have the expected time series data
      if (response.data && response.data['Time Series (Daily)']) {
        const result = response.data['Time Series (Daily)'];
        const stockData: StockDataPoint[] = [];
        
        for (const date in result) {
          if (date >= startDate && date <= endDate) {
            const data = result[date];
            stockData.push({
              date,
              timestamp: new Date(date).getTime(),
              open: parseFloat(data['1. open']),
              high: parseFloat(data['2. high']),
              low: parseFloat(data['3. low']),
              close: parseFloat(data['4. close']),
              volume: parseInt(data['5. volume']),
            });
          }
        }
        
        if (stockData.length > 0) {
          // Sort data points by date in ascending order
          return stockData.sort((a, b) => a.timestamp - b.timestamp);
        }
        
        console.warn('No data points found in the specified date range');
      } else {
        console.warn('Invalid API response format, expected Time Series (Daily)');
      }
      
      // If we reach here, something went wrong with the TIME_SERIES_DAILY API call
      // Try the TIME_SERIES_INTRADAY API as a fallback
      const intradayResponse = await axios.get(ALPHA_VANTAGE_API_URL, {
        params: {
          function: 'TIME_SERIES_INTRADAY',
          symbol,
          interval: '60min', // Using hourly data as fallback
          adjusted: 'true',
          outputsize: outputSize,
          datatype: dataType,
          apikey: ALPHA_VANTAGE_API_KEY
        },
        timeout: 10000 // 10 second timeout
      });
      
      if (intradayResponse.data && intradayResponse.data['Time Series (60min)']) {
        const result = intradayResponse.data['Time Series (60min)'];
        const stockData: StockDataPoint[] = [];
        const processedDates = new Set(); // To avoid duplicates
        
        for (const dateTime in result) {
          const date = dateTime.split(' ')[0]; // Extract the date part
          
          // Only process each date once and check if it's in our date range
          if (!processedDates.has(date) && date >= startDate && date <= endDate) {
            processedDates.add(date);
            const data = result[dateTime];
            
            stockData.push({
              date,
              timestamp: new Date(date).getTime(),
              open: parseFloat(data['1. open']),
              high: parseFloat(data['2. high']),
              low: parseFloat(data['3. low']),
              close: parseFloat(data['4. close']),
              volume: parseInt(data['5. volume']),
            });
          }
        }
        
        if (stockData.length > 0) {
          // Sort data points by date in ascending order
          return stockData.sort((a, b) => a.timestamp - b.timestamp);
        }
      }
      
      // If both API calls failed to provide usable data, return mock data
      console.warn('Failed to retrieve valid data from Alpha Vantage APIs, using mock data');
      return generateMockStockData(symbol, startDate, endDate);
      
    } catch (error) {
      console.error('Error fetching stock data:', error);
      return generateMockStockData(symbol, startDate, endDate);
    }
  } catch (error) {
    console.error('Unexpected error in fetchStockData:', error);
    return generateMockStockData(symbol, startDate, endDate);
  }
}

/**
 * Fetch real-time or intraday stock data from Alpha Vantage
 * 
 * @param symbol - Stock symbol (e.g., 'AAPL')
 * @param dataType - Optional: 'json' or 'csv' format for the response
 * @param interval - Optional: Time interval between data points (default: '1min')
 */
export async function fetchRealTimeStockData(
  symbol: string, 
  dataType: 'json' | 'csv' = 'json',
  interval: '1min' | '5min' | '15min' | '30min' | '60min' = '1min'
) {
  try {
    console.log(`Fetching real-time stock data for ${symbol} with interval ${interval}`);
    
    // For demo/testing, just return the last price if no valid API key
    if (!isApiKeyValid()) {
      console.warn('No valid API key available for real-time data, using mock data');
      return generateMockRealTimeData(symbol);
    }
    
    // Use the TIME_SERIES_INTRADAY endpoint for real-time data
    const response = await axios.get(ALPHA_VANTAGE_API_URL, {
      params: {
        function: 'TIME_SERIES_INTRADAY',
        symbol,
        interval,
        adjusted: 'true',
        outputsize: 'compact',
        datatype: dataType,
        apikey: ALPHA_VANTAGE_API_KEY
      },
      timeout: 10000
    });
    
    // Check for error responses
    if (response.data && response.data['Error Message']) {
      console.warn('Alpha Vantage API error:', response.data['Error Message']);
      return generateMockRealTimeData(symbol);
    }
    
    // Check for rate limiting
    if (response.data && response.data['Note'] && response.data['Note'].includes('API call frequency')) {
      console.warn('Alpha Vantage API rate limit reached:', response.data['Note']);
      return generateMockRealTimeData(symbol);
    }
    
    // Extract the time series data
    const timeSeriesKey = `Time Series (${interval})`;
    
    if (response.data && response.data[timeSeriesKey]) {
      const timeSeries = response.data[timeSeriesKey];
      const timestamps = Object.keys(timeSeries).sort((a, b) => new Date(b).getTime() - new Date(a).getTime());
      
      if (timestamps.length === 0) {
        console.warn('No intraday data available');
        return generateMockRealTimeData(symbol);
      }
      
      // Get the latest data point
      const latestTimestamp = timestamps[0];
      const latestData = timeSeries[latestTimestamp];
      
      return {
        symbol,
        timestamp: latestTimestamp,
        price: parseFloat(latestData['4. close']),
        change: 0, // We'll calculate this below if we have previous data
        changePercent: 0,
        open: parseFloat(latestData['1. open']),
        high: parseFloat(latestData['2. high']),
        low: parseFloat(latestData['3. low']),
        volume: parseInt(latestData['5. volume'])
      };
    }
    
    console.warn('Invalid API response format, using mock data');
    return generateMockRealTimeData(symbol);
  } catch (error) {
    console.error('Error fetching real-time stock data:', error);
    return generateMockRealTimeData(symbol);
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
  try {
    // Input validation with higher data requirements
    if (!historicalData || historicalData.length < 100) {
      console.warn('Insufficient data for prediction with 2-3 years of historical data. Minimum 100 data points required.');
      return [];
    }

    // Sort data chronologically to ensure consistency
    const sortedData = [...historicalData].sort((a, b) => {
      return new Date(a.date).getTime() - new Date(b.date).getTime();
    });

    // Log the data range being used
    const firstDate = new Date(sortedData[0].date);
    const lastDate = new Date(sortedData[sortedData.length - 1].date);
    const dataRangeMonths = (lastDate.getTime() - firstDate.getTime()) / (1000 * 60 * 60 * 24 * 30);
    
    console.log(`Training with ${sortedData.length} data points spanning ${dataRangeMonths.toFixed(1)} months from ${firstDate.toISOString().split('T')[0]} to ${lastDate.toISOString().split('T')[0]}`);

    // Initialize predictions array
    let predictions: PredictionPoint[] = [];

    // Verify that the last date in the data is not too far in the future
    const today = new Date();
    if (lastDate.getTime() > today.getTime() + 24 * 60 * 60 * 1000) {
      console.warn('Last date in data is too far in the future.');
      return [];
    }

    // Use appropriate model for prediction
    const modelType = algorithm.toLowerCase();
    console.log(`🔮 Using prediction algorithm: ${modelType} with extended historical data`);
    
    // Use the requested algorithm without fallbacks
    if (modelType === 'lstm') {
      console.log('Creating LSTM model for prediction with extended historical data');
      // Create and train LSTM model using the model factory
      const lstmModel = createModel('lstm') as LSTMModel;
      await lstmModel.train(sortedData);
      predictions = await lstmModel.predict(sortedData, predictionDays);
      // Add algorithm info to predictions
      predictions = predictions.map(p => ({...p, algorithmUsed: 'lstm'}));
    } else if (modelType === 'transformer') {
      console.log('Creating Transformer model for prediction with extended historical data');
      // Create Transformer model using the model factory
      const transformerModel = createModel('transformer') as TransformerModel;
      await transformerModel.train(sortedData);
      predictions = await transformerModel.predict(sortedData, predictionDays);
      // Add algorithm info to predictions
      predictions = predictions.map(p => ({...p, algorithmUsed: 'transformer'}));
    } else if (modelType === 'arima') {
      console.log('Creating ARIMA model for prediction with extended historical data');
      // Use ARIMA model for prediction
      predictions = await generateARIMAForecast(sortedData, predictionDays);
      // Add algorithm info to predictions
      predictions = predictions.map(p => ({...p, algorithmUsed: 'arima'}));
    } else if (modelType === 'prophet') {
      console.log('Creating Prophet model for prediction');
      // Previously used ARIMA as fallback, now use empty predictions to trigger error handling
      return [];
    } else if (modelType === 'tddm') {
      console.log('Creating TDDM model for prediction with extended historical data');
      // Create TDDM model using the model factory
      const tddmModel = createModel('tddm') as TDDMModel;
      await tddmModel.train(sortedData);
      predictions = await tddmModel.predict(sortedData, predictionDays);
      // Add algorithm info to predictions
      predictions = predictions.map(p => ({...p, algorithmUsed: 'tddm'}));
    } else if (modelType === 'gan') {
      console.log('Creating GAN model for prediction');
      // Previously used LSTM as fallback, now use empty predictions to trigger error handling
      return [];
    } else if (modelType === 'xgboost') {
      console.log('Creating XGBoost model for prediction with extended historical data');
      // Create XGBoost model using the model factory
      const xgboostModel = createModel('xgboost') as XGBoostModel;
      await xgboostModel.train(sortedData);
      predictions = await xgboostModel.predict(sortedData, predictionDays);
      // Add algorithm info to predictions
      predictions = predictions.map(p => ({...p, algorithmUsed: 'xgboost'}));
    } else if (modelType === 'cnnlstm') {
      console.log('Creating CNN-LSTM model for prediction with extended historical data');
      // Create CNN-LSTM model using the model factory
      const cnnlstmModel = createModel('cnnlstm') as CNNLSTMModel;
      await cnnlstmModel.train(sortedData);
      predictions = await cnnlstmModel.predict(sortedData, predictionDays);
      // Add algorithm info to predictions
      predictions = predictions.map(p => ({...p, algorithmUsed: 'cnnlstm'}));
    } else if (modelType === 'ensemble') {
      console.log('Creating Ensemble model for prediction with extended historical data');
      // Create Ensemble model using the model factory
      const ensembleModel = createModel('ensemble') as EnsembleModel;
      await ensembleModel.train(sortedData);
      predictions = await ensembleModel.predict(sortedData, predictionDays);
      // Add algorithm info to predictions
      predictions = predictions.map(p => ({...p, algorithmUsed: 'ensemble'}));
    } else if (modelType === 'movingaverage') {
      console.log('Using Moving Average model for prediction with extended historical data');
      predictions = await generateMovingAverageForecast(sortedData, predictionDays);
      // Add algorithm info to predictions
      predictions = predictions.map(p => ({...p, algorithmUsed: 'movingaverage'}));
    } else {
      console.warn(`Unknown algorithm: ${algorithm}. Cannot make prediction.`);
      return [];
    }

    // Validate predictions
    if (predictions && predictions.length > 0 && isFinite(predictions[0].price)) {
      console.log(`Successfully generated ${predictions.length} predictions using ${modelType} with extended historical data`);
      return predictions;
    }
    
    // If we got here, the predictions are invalid
    console.warn(`Invalid predictions from ${modelType} with extended historical data. Returning empty array.`);
    return [];

  } catch (error) {
    console.error('Error in stock price prediction:', error);
    return [];
  }
}

/**
 * Generate forecast based on moving averages
 * This implements a real SMA-based forecasting model
 */
export async function generateMovingAverageForecast(historicalData: StockDataPoint[], days: number): Promise<PredictionPoint[]> {
  const predictions: PredictionPoint[] = [];
  const lastPrice = historicalData[historicalData.length - 1].close;
  const lastDate = new Date(historicalData[historicalData.length - 1].date);
  
  // Calculate multiple moving averages for better prediction
  const shortWindow = Math.min(10, Math.floor(historicalData.length / 3));
  const mediumWindow = Math.min(20, Math.floor(historicalData.length / 2));
  const longWindow = Math.min(50, Math.floor(historicalData.length * 0.8));
  
  const shortMA = calculateSMA(historicalData, shortWindow) || lastPrice;
  const mediumMA = calculateSMA(historicalData, mediumWindow) || lastPrice;
  const longMA = calculateSMA(historicalData, longWindow) || lastPrice;
  
  console.log('Moving Average values:', { shortMA, mediumMA, longMA, lastPrice });
  
  // Calculate trend based on multiple moving average crossovers
  // Short-term trend (more weight)
  const shortTrend = mediumMA !== 0 ? (shortMA - mediumMA) / mediumMA : 0;
  // Long-term trend (less weight)
  const longTrend = longMA !== 0 ? (mediumMA - longMA) / longMA : 0;
  
  // Combined trend with weights - ensure it's a finite value
  const rawTrend = (shortTrend * 0.7) + (longTrend * 0.3);
  let trend = isFinite(rawTrend) ? rawTrend : 0;
  
  // Constrain trend to reasonable values
  trend = Math.max(-0.005, Math.min(0.005, trend));
  
  // If trend is negative, make it more conservative
  if (trend < 0) {
    trend = trend * 0.5;
  }
  
  console.log('Calculated trend:', trend);
  
  // Calculate volatility from historical data
  const volatility = calculateVolatility(historicalData);
  
  // Calculate RSI to determine overbought/oversold conditions
  const rsi = calculateRSI(historicalData);
  
  // Adjust trend based on RSI (mean reversion effects)
  let adjustedTrend = trend;
  if (rsi > 70) {
    // Overbought condition, expect some reversion
    adjustedTrend = trend * 0.5;
  } else if (rsi < 30) {
    // Oversold condition, expect some reversion
    adjustedTrend = trend * 1.5;
  }
  
  // Project future prices
  let currentPrice = lastPrice;
  
  for (let i = 1; i <= days; i++) {
    const predictionDate = new Date(lastDate);
    predictionDate.setDate(lastDate.getDate() + i);
    
    // Apply trend with decreasing certainty over time
    const trendFactor = adjustedTrend * Math.exp(-i/60); // Slower decay
    
    // Add smaller randomness based on historical volatility 
    // (increased uncertainty over time)
    const randomComponent = generateNormalRandom() * volatility * Math.sqrt(i/15) * 0.5; // Reduced randomness
    
    // Update price with trend and randomness - ensure it's a finite value
    const priceChange = trendFactor + randomComponent;
    currentPrice = currentPrice * (1 + (isFinite(priceChange) ? priceChange : 0));
    
    // Ensure price doesn't deviate too much from moving averages (mean reversion)
    const meanReversionTarget = (shortMA + mediumMA + lastPrice) / 3;
    const meanReversionStrength = 0.1 * (i / days); // Increases with time
    currentPrice = currentPrice * (1 - meanReversionStrength) + meanReversionTarget * meanReversionStrength;
    
    // Ensure price doesn't drop too much from the last historical price
    const minPrice = lastPrice * Math.max(0.85, 1 - (i * 0.005)); // Max 15% drop or 0.5% per day
    currentPrice = Math.max(minPrice, currentPrice);
    
    // Ensure price is a valid positive number
    currentPrice = isFinite(currentPrice) ? Math.max(0.01, currentPrice) : lastPrice;
    
    // Calculate confidence interval (grows with time)
    const confIntervalRaw = volatility * Math.sqrt(i/10) * 1.96; // 95% confidence
    const confInterval = isFinite(confIntervalRaw) ? confIntervalRaw : 0.01;
    
    const upperBound = isFinite(currentPrice * (1 + confInterval)) ? 
        currentPrice * (1 + confInterval) : currentPrice * 1.01;
    
    const lowerBound = isFinite(currentPrice * (1 - confInterval)) ?
        Math.max(minPrice * 0.95, currentPrice * (1 - confInterval)) : currentPrice * 0.99;
    
    predictions.push({
      date: predictionDate.toISOString().split('T')[0],
      price: currentPrice,
      upper: upperBound,
      lower: lowerBound,
      algorithmUsed: 'movingaverage'
    });
  }
  
  return predictions;
}

/**
 * Generate LSTM forecast using TensorFlow.js
 */
export async function generateLSTMForecast(historicalData: StockDataPoint[], days: number): Promise<PredictionPoint[]> {
  // Create LSTM model with fast training settings
  const lstmModel = createModel('lstm', {
    // These parameters are already set in the createModel function
    // which now uses our fast training configuration
  });
  
  try {
    console.log('Training LSTM model with fast training configuration');
    // Train the model with historical data
    await lstmModel.train(historicalData);
    
    // Generate predictions using the trained model
    return await lstmModel.predict(historicalData, days);
  } catch (error) {
    console.error('Error in LSTM prediction:', error);
    // Fallback to simpler prediction method if LSTM fails
    console.log('Falling back to simpler prediction method');
    return generateMovingAverageForecast(historicalData, days);
  }
}

/**
 * Generate Transformer forecast using TensorFlow.js
 */
export async function generateTransformerForecast(historicalData: StockDataPoint[], days: number): Promise<PredictionPoint[]> {
  // Create Transformer model using the model factory
  const transformerModel = createModel('transformer');
  
  try {
    // Train the model with historical data
    await transformerModel.train(historicalData);
    
    // Generate predictions using the trained model
    return await transformerModel.predict(historicalData, days);
  } catch (error) {
    console.error('Error in Transformer prediction:', error);
    throw error;
  }
}

/**
 * Generate ARIMA forecast
 * Implements a simplified ARIMA model since full ARIMA is challenging in JavaScript
 */
export async function generateARIMAForecast(historicalData: StockDataPoint[], days: number): Promise<PredictionPoint[]> {
  const predictions: PredictionPoint[] = [];
  const lastPrice = historicalData[historicalData.length - 1].close;
  const lastDate = new Date(historicalData[historicalData.length - 1].date);
  
  // Extract price series
  const prices = historicalData.map(d => d.close);
  
  // Calculate price differences for AR component
  const priceDiffs = [];
  for (let i = 1; i < prices.length; i++) {
    priceDiffs.push(prices[i] - prices[i-1]);
  }
  
  // AR parameters: we'll use yule-walker method to find coefficients
  const arLag = 5; // Using 5-day AR model
  const arCoefficients = fitARModel(priceDiffs, arLag);
  
  // Mean of differences for constant term
  const meanDiff = priceDiffs.reduce((sum, diff) => sum + diff, 0) / priceDiffs.length;
  
  // Calculate volatility of the differenced series
  const diffVolatility = calculateStdDev(priceDiffs);
  
  // Track recent differences for AR model
  const recentDiffs = priceDiffs.slice(-arLag);
  
  // Project future prices
  let currentPrice = lastPrice;
  
  for (let i = 1; i <= days; i++) {
    const predictionDate = new Date(lastDate);
    predictionDate.setDate(lastDate.getDate() + i);
    
    // Calculate AR component
    let arComponent = meanDiff; // Constant term
    for (let j = 0; j < Math.min(arLag, recentDiffs.length); j++) {
      arComponent += arCoefficients[j] * recentDiffs[recentDiffs.length - 1 - j];
    }
    
    // Add moving average component (MA)
    const maComponent = 0.3 * (generateNormalRandom() * diffVolatility);
    
    // Add error component with increasing uncertainty over time
    const errorComponent = generateNormalRandom() * diffVolatility * Math.sqrt(i/5);
    
    // Calculate price difference
    const priceDiff = arComponent + maComponent + errorComponent;
    
    // Update price
    currentPrice += priceDiff;
    currentPrice = Math.max(currentPrice, 0.01); // Ensure positive price
    
    // Update recent diffs for next prediction
    recentDiffs.push(priceDiff);
    if (recentDiffs.length > arLag) {
      recentDiffs.shift();
    }
    
    // Calculate confidence interval
    const confInterval = diffVolatility * Math.sqrt(i) * 1.96;
    
    predictions.push({
      date: predictionDate.toISOString().split('T')[0],
      price: currentPrice,
      upper: currentPrice + confInterval * Math.sqrt(i),
      lower: Math.max(0.01, currentPrice - confInterval * Math.sqrt(i)),
      algorithmUsed: 'arima',
    });
  }
  
  return predictions;
}

// XGBoost feature extraction has been moved to app/models/XGBoostModel.ts

// XGBoost helper functions have been moved to app/models/XGBoostModel.ts

/**
 * Helper function to clamp a value between min and max
 */
function clamp(value: number, min: number, max: number): number {
  if (!isFinite(value)) return (min + max) / 2;
  return Math.max(min, Math.min(max, value));
}

/**
 * Calculate Simple Moving Average
 */
function calculateSMA(data: StockDataPoint[], window: number): number {
  if (!data || data.length === 0 || window <= 0) {
    return 0;
  }
  
  const effectiveWindow = Math.min(window, data.length);
  const windowData = data.slice(Math.max(0, data.length - effectiveWindow));
  
  if (windowData.length === 0) {
    return data[data.length - 1]?.close || 0;
  }
  
  const sum = windowData.reduce((sum, point) => sum + (isFinite(point.close) ? point.close : 0), 0);
  const result = sum / windowData.length;
  
  return isFinite(result) ? result : 0;
}

/**
 * Calculate price momentum
 */
function calculateMomentum(data: StockDataPoint[], period: number): number {
  if (!data || data.length <= period || period <= 0) {
    return 0;
  }
  
  const oldPriceIdx = Math.max(0, data.length - period - 1);
  const currentIdx = data.length - 1;
  
  const oldPrice = data[oldPriceIdx]?.close;
  const currentPrice = data[currentIdx]?.close;
  
  // Check for valid prices
  if (!isFinite(oldPrice) || !isFinite(currentPrice) || oldPrice === 0) {
    return 0;
  }
  
  const momentum = (currentPrice - oldPrice) / oldPrice;
  
  // Cap very large values to prevent numerical issues
  if (momentum > 10) return 10;
  if (momentum < -10) return -10;
  
  return isFinite(momentum) ? momentum : 0;
}

/**
 * Calculate Relative Strength Index
 */
function calculateRSI(data: StockDataPoint[], period: number = 14): number {
  if (!data || data.length < period + 1) {
    return 50;
  }
  
  let gains = 0;
  let losses = 0;
  
  for (let i = data.length - period; i < data.length; i++) {
    if (i <= 0) continue;
    
    const change = data[i].close - data[i - 1].close;
    if (!isFinite(change)) continue;
    
    if (change >= 0) {
      gains += change;
    } else {
      losses -= change;
    }
  }
  
  if (losses === 0) return 100;
  if (gains === 0) return 0;
  
  const rs = gains / losses;
  const rsi = 100 - (100 / (1 + rs));
  
  return isFinite(rsi) ? rsi : 50;
}

/**
 * Calculate volatility as standard deviation of returns
 */
function calculateVolatility(data: StockDataPoint[]): number {
  if (!data || data.length < 2) return 0.01;
  
  const returns = [];
  for (let i = 1; i < data.length; i++) {
    const prevClose = data[i-1].close;
    if (prevClose === 0 || !isFinite(prevClose)) continue;
    
    const currentReturn = (data[i].close - prevClose) / prevClose;
    if (isFinite(currentReturn)) {
      returns.push(currentReturn);
    }
  }
  
  if (returns.length === 0) return 0.01;
  
  const meanReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - meanReturn, 2), 0) / returns.length;
  
  const volatility = Math.sqrt(variance);
  return isFinite(volatility) ? volatility : 0.01;
}

/**
 * Calculate standard deviation of a series
 */
function calculateStdDev(data: number[]): number {
  if (data.length < 2) return 0.01;
  
  const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
  const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
  
  return Math.sqrt(variance);
}

/**
 * Generate normally distributed random number
 */
function generateNormalRandom(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  
  const result = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  // Ensure the result is a valid finite number
  return isFinite(result) ? result : 0;
}

/**
 * Fit an AR model using Yule-Walker equations
 */
function fitARModel(data: number[], lag: number): number[] {
  // Calculate autocorrelation coefficients
  const acf = [];
  const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
  
  for (let i = 0; i <= lag; i++) {
    let numerator = 0;
    let denominator = 0;
    
    for (let j = 0; j < data.length - i; j++) {
      numerator += (data[j] - mean) * (data[j + i] - mean);
    }
    
    for (let j = 0; j < data.length; j++) {
      denominator += Math.pow(data[j] - mean, 2);
    }
    
    acf.push(numerator / denominator);
  }
  
  // Solve Yule-Walker equations using basic linear algebra
  const R = [];
  for (let i = 0; i < lag; i++) {
    R.push(acf.slice(0, lag));
  }
  
  const r = acf.slice(1, lag + 1);
  
  // Very simple solver for demonstration - for real applications use a proper linear algebra library
  const coefficients = [];
  for (let i = 0; i < lag; i++) {
    coefficients.push(r[i] / acf[0]); // Simplified approximation
  }
  
  return coefficients;
}

/**
 * Generate mock stock data for demonstration purposes
 */
function generateMockStockData(
  symbol: string,
  startDate: string,
  endDate: string
): StockDataPoint[] {
  console.log(`Generating mock data for ${symbol} from ${startDate} to ${endDate}`);
  
  try {
    const startTimestamp = new Date(startDate).getTime();
    const endTimestamp = new Date(endDate).getTime();
    
    // Validate dates - if invalid, use reasonable defaults
    if (!isFinite(startTimestamp) || !isFinite(endTimestamp) || startTimestamp > endTimestamp) {
      const today = new Date();
      const oneMonthAgo = new Date();
      oneMonthAgo.setMonth(today.getMonth() - 1);
      
      const validStartTimestamp = oneMonthAgo.getTime();
      const validEndTimestamp = today.getTime();
      
      console.warn(`Invalid date range: ${startDate} to ${endDate}, using last month instead`);
      return generateMockStockData(
        symbol, 
        oneMonthAgo.toISOString().split('T')[0], 
        today.toISOString().split('T')[0]
      );
    }
    
    const dayDuration = 24 * 60 * 60 * 1000; // 1 day in milliseconds
    const days = Math.ceil((endTimestamp - startTimestamp) / dayDuration);
    
    // Cap at reasonable number of days to prevent excessive data
    const maxDays = 365 * 2; // 2 years max
    const effectiveDays = Math.min(Math.max(1, days), maxDays);
    
    const basePrice = symbol === 'AAPL' ? 150 : 
                     symbol === 'MSFT' ? 280 : 
                     symbol === 'GOOGL' ? 2500 : 
                     symbol === 'AMZN' ? 3300 : 
                     symbol === 'TSLA' ? 800 : 
                     symbol === 'FB' ? 330 : 
                     symbol === 'META' ? 330 :
                     symbol === 'NVDA' ? 700 : 
                     symbol === 'JPM' ? 150 : 100;
    
    const data: StockDataPoint[] = [];
    let currentPrice = basePrice;
    
    for (let i = 0; i < effectiveDays; i++) {
      const timestamp = startTimestamp + (i * dayDuration);
      const date = new Date(timestamp).toISOString().split('T')[0];
      
      // Add some random fluctuation - ensure it's reasonable
      const randomFactor = Math.random();
      const changePercent = (randomFactor - 0.48) * 3; // Slightly biased towards up
      
      // Ensure price change is reasonable and price stays positive
      currentPrice = Math.max(currentPrice * (1 + changePercent / 100), 0.1);
      
      // Add some patterns to make it look more realistic
      if (i % 30 === 0) {
        currentPrice = currentPrice * (1 + (Math.random() > 0.5 ? 1 : -1) * Math.random() * 0.05);
      }
      
      // Keep price within reasonable bounds
      if (currentPrice > basePrice * 5) {
        currentPrice = basePrice * (2 + Math.random());
      } else if (currentPrice < basePrice * 0.2) {
        currentPrice = basePrice * (0.5 + Math.random() * 0.5);
      }
      
      const volume = Math.floor(Math.random() * 10000000) + 1000000;
      
      // Calculate related values consistently
      const closePrice = currentPrice;
      const openPrice = closePrice * (1 - Math.random() * 0.01);
      const highPrice = closePrice * (1 + Math.random() * 0.015);
      const lowPrice = openPrice * (1 - Math.random() * 0.015);
      
      // Ensure all values are valid and properly ordered (low ≤ open, close ≤ high)
      const safeOpen = isFinite(openPrice) ? openPrice : closePrice;
      const safeHigh = isFinite(highPrice) ? Math.max(highPrice, safeOpen, closePrice) : closePrice * 1.01;
      const safeLow = isFinite(lowPrice) ? Math.min(lowPrice, safeOpen, closePrice) : closePrice * 0.99;
      const safeClose = isFinite(closePrice) ? closePrice : basePrice;
      
      data.push({
        date,
        timestamp,
        open: safeOpen,
        high: safeHigh,
        low: safeLow,
        close: safeClose,
        volume
      });
    }
    
    return data;
  } catch (error) {
    console.error("Error generating mock stock data:", error);
    // Return minimal valid dataset in case of error
    const today = new Date();
    const yesterday = new Date();
    yesterday.setDate(today.getDate() - 1);
    
    return [
      {
        date: yesterday.toISOString().split('T')[0],
        timestamp: yesterday.getTime(),
        open: 100,
        high: 105,
        low: 95,
        close: 102,
        volume: 1000000
      },
      {
        date: today.toISOString().split('T')[0],
        timestamp: today.getTime(),
        open: 102,
        high: 107,
        low: 101,
        close: 106,
        volume: 1100000
      }
    ];
  }
}

/**
 * Generate mock real-time stock data
 */
function generateMockRealTimeData(symbol: string) {
  console.log(`Generating mock real-time data for ${symbol}`);
  
  try {
    if (!symbol || typeof symbol !== 'string') {
      console.warn('Invalid symbol provided to generateMockRealTimeData');
      symbol = 'UNKNOWN';
    }
    
    // Define base prices for known symbols
    const basePrice = symbol === 'AAPL' ? 150 : 
                     symbol === 'MSFT' ? 280 : 
                     symbol === 'GOOGL' ? 2500 : 
                     symbol === 'AMZN' ? 3300 : 
                     symbol === 'TSLA' ? 800 : 
                     symbol === 'META' ? 330 : 
                     symbol === 'FB' ? 330 : 
                     symbol === 'NVDA' ? 700 : 
                     symbol === 'JPM' ? 150 : 100;
    
    // Generate random price changes
    const change = parseFloat((Math.random() * 10 - 5).toFixed(2));
    const changePercent = parseFloat((Math.random() * 5 - 2.5).toFixed(2));
    
    // Calculate price based on change
    const price = Math.max(0.01, basePrice + parseFloat(change.toString()));
    
    return {
      symbol,
      price: isFinite(price) ? price : basePrice,
      change: isFinite(change) ? change : 0,
      changePercent: isFinite(changePercent) ? changePercent : 0,
      volume: Math.floor(Math.random() * 10000000),
    };
  } catch (error) {
    console.error("Error generating mock real-time data:", error);
    // Return safe fallback data
    return {
      symbol: symbol || 'UNKNOWN',
      price: 100,
      change: 0,
      changePercent: 0,
      volume: 1000000,
    };
  }
}

/**
 * Global error handler for any API function
 * This wrapper ensures that no server errors are thrown to the client
 * @param apiFunction The API function to call
 * @param fallbackValue Fallback value if the function fails
 * @param args Arguments to pass to the function
 */
export async function safeApiCall<T>(apiFunction: Function, fallbackValue: T, ...args: any[]): Promise<T> {
  try {
    // Call the API function with the provided arguments
    const result = await apiFunction(...args);
    return result;
  } catch (error) {
    console.error(`API call failed: ${apiFunction.name || 'unknown function'}`, error);
    return fallbackValue;
  }
}

/**
 * Safe wrapper for fetching stock data that handles errors gracefully
 */
export async function safeGetStockData(
  symbol: string,
  startDate: string,
  endDate: string,
  outputSize: 'compact' | 'full' = 'compact',
  dataType: 'json' | 'csv' = 'json'
): Promise<StockDataPoint[]> {
  try {
    // Validate dates before making API call
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];
    
    let adjustedEndDate = endDate;
    let adjustedStartDate = startDate;
    
    // Ensure endDate is not in the future
    if (endDate > yesterdayStr) {
      console.warn(`[safeGetStockData] End date ${endDate} is in the future, adjusting to yesterday (${yesterdayStr})`);
      adjustedEndDate = yesterdayStr;
    }
    
    // Ensure startDate is not after endDate
    if (startDate > adjustedEndDate) {
      console.warn(`[safeGetStockData] Start date ${startDate} is after end date ${adjustedEndDate}, adjusting start date`);
      const oneYearBefore = new Date(adjustedEndDate);
      oneYearBefore.setFullYear(oneYearBefore.getFullYear() - 1);
      adjustedStartDate = oneYearBefore.toISOString().split('T')[0];
    }
    
    console.log(`[safeGetStockData] Fetching data for ${symbol} from ${adjustedStartDate} to ${adjustedEndDate}`);
    const data = await fetchStockData(symbol, adjustedStartDate, adjustedEndDate, outputSize, dataType);
    
    if (!data || data.length === 0) {
      console.warn(`[safeGetStockData] No data returned for ${symbol}, using mock data`);
      return generateMockStockData(symbol, adjustedStartDate, adjustedEndDate);
    }
    
    return data;
  } catch (error) {
    console.error(`[safeGetStockData] Error fetching data for ${symbol}:`, error);
    return generateMockStockData(symbol, startDate, endDate);
  }
}

/**
 * Safe wrapper for fetchRealTimeStockData
 */
export async function safeGetRealTimeData(
  symbol: string,
  dataType: 'json' | 'csv' = 'json',
  interval: '1min' | '5min' | '15min' | '30min' | '60min' = '1min'
) {
  const fallback = generateMockRealTimeData(symbol);
  return await safeApiCall(fetchRealTimeStockData, fallback, symbol, dataType, interval);
}

/**
 * Safe wrapper for predictStockPrice
 */
export async function safePredictStockPrice(
  symbol: string,
  historicalData: StockDataPoint[],
  algorithm: string,
  predictionDays: number = 30
): Promise<{ predictions: PredictionPoint[], actualAlgorithm: string }> {
  try {
    console.log(`[safePredictStockPrice] Generating predictions for ${symbol} using ${algorithm} with ${historicalData?.length || 0} historical data points`);
    
    // Validate input data - require more data for better predictions
    if (!historicalData || historicalData.length < 100) {
      console.warn(`[safePredictStockPrice] Insufficient historical data (${historicalData?.length || 0} points, minimum 100 required), using fallback`);
      const fallbackPredictions = createFallbackPrediction(historicalData || [], predictionDays);
      return { predictions: fallbackPredictions, actualAlgorithm: 'fallback' };
    }
    
    // Call the actual prediction function with timeout protection
    let predictions: PredictionPoint[] = [];
    let actualAlgorithm = algorithm;
    
    try {
      // Add a timeout to prevent hanging on model training - increased for CNN-LSTM model
      const timeoutMs = algorithm.toLowerCase() === 'cnnlstm' ? 60000 : 45000; // 60 seconds for CNN-LSTM, 45 for others
      console.log(`[safePredictStockPrice] Setting timeout of ${timeoutMs/1000} seconds for ${algorithm}`);
      
      const timeoutPromise = new Promise<PredictionPoint[]>((_, reject) => {
        setTimeout(() => reject(new Error(`Prediction timeout after ${timeoutMs/1000} seconds`)), timeoutMs);
      });
      
      // Race between the actual prediction and the timeout
      predictions = await Promise.race([
        predictStockPrice(symbol, historicalData, algorithm, predictionDays),
        timeoutPromise
      ]);
      
      // Check if the predictions have an algorithmUsed field
      if (predictions.length > 0 && predictions[0].algorithmUsed) {
        actualAlgorithm = predictions[0].algorithmUsed;
      }
      
      console.log(`[safePredictStockPrice] Model prediction completed with ${predictions.length} points using ${actualAlgorithm}`);
      
      // Ensure first prediction starts from last historical price for smooth transition
      if (predictions.length > 0 && historicalData.length > 0) {
        const lastHistoricalPoint = historicalData[historicalData.length - 1];
        const firstPredictionPoint = predictions[0];
        
        // Adjust first prediction to start from last historical price
        predictions[0] = {
          ...firstPredictionPoint,
          price: lastHistoricalPoint.close,
          lower: lastHistoricalPoint.close * 0.99, // 1% lower
          upper: lastHistoricalPoint.close * 1.01  // 1% higher
        };
      }
      
      console.log(`[safePredictStockPrice] Successfully generated ${predictions.length} predictions using ${actualAlgorithm}`);
      return { predictions, actualAlgorithm };
    } catch (modelError) {
      console.error(`[safePredictStockPrice] Error in model prediction:`, modelError);
      console.warn(`[safePredictStockPrice] Falling back to simple prediction for ${algorithm}`);
      const fallbackPredictions = createFallbackPrediction(historicalData, predictionDays);
      return { predictions: fallbackPredictions, actualAlgorithm: 'fallback' };
    }
  } catch (error) {
    console.error(`[safePredictStockPrice] Error generating predictions:`, error);
    // Create a minimal fallback prediction if needed
    const fallbackPredictions = createFallbackPrediction(historicalData || [], predictionDays);
    return { predictions: fallbackPredictions, actualAlgorithm: 'fallback' };
  }
}

/**
 * Create a simple fallback prediction when the main prediction fails
 */
function createFallbackPrediction(historicalData: StockDataPoint[], days: number): PredictionPoint[] {
  const predictions: PredictionPoint[] = [];
  
  // Get last price or a default value
  const lastPrice = historicalData && historicalData.length > 0 && isFinite(historicalData[historicalData.length - 1].close) 
    ? historicalData[historicalData.length - 1].close 
    : 100;
    
  // Get last date or current date
  const lastDate = historicalData && historicalData.length > 0
    ? new Date(historicalData[historicalData.length - 1].date)
    : new Date();
  
  // Calculate a more stable trend from historical data
  let trend = 0.0001; // Small positive default trend (0.01% daily)
  
  if (historicalData && historicalData.length >= 20) {
    // Use longer period for trend calculation (20 days)
    const recentData = historicalData.slice(-20);
    
    // Calculate moving averages to smooth out noise
    const shortMA = calculateSMA(recentData, 5);
    const longMA = calculateSMA(recentData, 20);
    
    // Use moving averages to determine trend direction
    if (shortMA && longMA) {
      trend = (shortMA - longMA) / longMA / 20; // Normalize to daily trend
    } else {
      // Fallback to simple calculation if MA fails
      const firstPrice = recentData[0].close;
      const lastRecentPrice = recentData[recentData.length - 1].close;
      
      // Calculate average daily percentage change
      trend = (lastRecentPrice / firstPrice - 1) / recentData.length;
    }
    
    // Ensure trend is reasonable - much tighter bounds
    trend = Math.max(-0.002, Math.min(0.002, trend));
    
    // Always ensure a slight upward bias
    if (trend < 0) {
      trend = trend * 0.3; // Significantly reduce negative trends
    }
  }
  
  console.log('Using trend for prediction:', trend);
  
  // Calculate price momentum (acceleration/deceleration of price changes)
  let momentum = 0;
  if (historicalData && historicalData.length >= 10) {
    const recentPrices = historicalData.slice(-10).map(d => d.close);
    const changes = [];
    for (let i = 1; i < recentPrices.length; i++) {
      changes.push(recentPrices[i] / recentPrices[i-1] - 1);
    }
    
    // Calculate if momentum is increasing or decreasing
    let increasingCount = 0;
    let decreasingCount = 0;
    for (let i = 1; i < changes.length; i++) {
      if (changes[i] > changes[i-1]) increasingCount++;
      if (changes[i] < changes[i-1]) decreasingCount++;
    }
    
    // Set momentum based on recent price change direction
    momentum = (increasingCount - decreasingCount) / (increasingCount + decreasingCount) * 0.0005;
  }
  
  // Add some volatility based on historical data - but keep it very small
  let volatility = 0.005; // Default 0.5% daily volatility
  if (historicalData && historicalData.length >= 10) {
    const recentPrices = historicalData.slice(-10).map(d => d.close);
    const avgPrice = recentPrices.reduce((sum, price) => sum + price, 0) / recentPrices.length;
    
    // Calculate standard deviation
    const squaredDiffs = recentPrices.map(price => Math.pow(price - avgPrice, 2));
    const avgSquaredDiff = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / squaredDiffs.length;
    volatility = Math.sqrt(avgSquaredDiff) / avgPrice;
    
    // Ensure volatility is very reasonable (0.2% to 1%)
    volatility = Math.max(0.002, Math.min(0.01, volatility));
  }
  
  // Generate smooth predictions with minimal randomness
  let currentPrice = lastPrice;
  
  // Calculate the overall market direction from the last 30-60 days if available
  let marketDirection = 0;
  if (historicalData && historicalData.length >= 60) {
    const longTermData = historicalData.slice(-60);
    const firstLongTermPrice = longTermData[0].close;
    const lastLongTermPrice = longTermData[longTermData.length - 1].close;
    marketDirection = (lastLongTermPrice / firstLongTermPrice - 1) / 60; // Daily rate
    marketDirection = Math.max(-0.001, Math.min(0.001, marketDirection)); // Constrain it
  }
  
  // Generate predictions
  for (let i = 1; i <= days; i++) {
    const predictionDate = new Date(lastDate);
    predictionDate.setDate(lastDate.getDate() + i);
    
    // Calculate base trend with momentum effect (trend accelerates/decelerates)
    const adjustedTrend = trend + (momentum * i / 30);
    
    // Apply market direction with increasing weight over time
    const marketWeight = Math.min(0.7, i / (days * 1.5));
    const combinedTrend = adjustedTrend * (1 - marketWeight) + marketDirection * marketWeight;
    
    // Add minimal randomness that decreases over time (more certain short-term, less random)
    const randomFactor = generateNormalRandom() * volatility * Math.exp(-i/30) * 0.3;
    
    // Calculate price change with dampening for longer predictions
    const priceChange = combinedTrend + randomFactor;
    
    // Update price with compound effect
    currentPrice = currentPrice * (1 + priceChange);
    
    // Apply mean reversion to prevent extreme predictions
    // Price gradually reverts to a moving average over time
    if (i > 5) {
      const meanReversionStrength = Math.min(0.3, (i - 5) / (days * 2));
      const targetPrice = lastPrice * (1 + trend * i); // Simple linear projection
      currentPrice = currentPrice * (1 - meanReversionStrength) + targetPrice * meanReversionStrength;
    }
    
    // Ensure price remains reasonable
    // Allow at most 20% total change over the prediction period
    const maxChange = 1 + (0.2 * i / days); // Max 20% increase
    const minChange = 1 - (0.1 * i / days); // Max 10% decrease
    
    const maxPrice = lastPrice * maxChange;
    const minPrice = lastPrice * minChange;
    
    currentPrice = Math.min(maxPrice, Math.max(minPrice, currentPrice));
    
    // Calculate confidence interval (grows with time)
    const dayFactor = Math.sqrt(i / 10);
    const confidenceInterval = volatility * dayFactor;
    
    predictions.push({
      date: predictionDate.toISOString().split('T')[0],
      price: currentPrice,
      upper: currentPrice * (1 + confidenceInterval),
      lower: currentPrice * (1 - confidenceInterval),
      algorithmUsed: 'fallback',
    });
  }
  
  console.log('Created fallback prediction with trend:', trend, 'volatility:', volatility);
  
  return predictions;
} 