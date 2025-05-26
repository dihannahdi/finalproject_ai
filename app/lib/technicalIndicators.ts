import { StockDataPoint } from './api';

// Simple Moving Average (SMA)
export function calculateSMA(data: number[], period: number): number[] {
  const result: number[] = [];
  
  if (data.length < period) {
    return new Array(data.length).fill(null);
  }
  
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += data[i];
  }
  
  result.push(sum / period);
  
  for (let i = period; i < data.length; i++) {
    sum = sum - data[i - period] + data[i];
    result.push(sum / period);
  }
  
  const padding = new Array(period - 1).fill(null);
  return [...padding, ...result];
}

// Exponential Moving Average (EMA)
export function calculateEMA(data: number[], period: number): number[] {
  const result: number[] = [];
  const multiplier = 2 / (period + 1);
  
  if (data.length < period) {
    return new Array(data.length).fill(null);
  }
  
  // Start with SMA
  let ema = data.slice(0, period).reduce((sum, price) => sum + price, 0) / period;
  
  const padding = new Array(period - 1).fill(null);
  result.push(ema);
  
  for (let i = period; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
    result.push(ema);
  }
  
  return [...padding, ...result];
}

// Relative Strength Index (RSI)
export function calculateRSI(data: number[], period: number = 14): number[] {
  const result: number[] = [];
  
  if (data.length < period + 1) {
    return new Array(data.length).fill(null);
  }
  
  // Calculate price changes
  const changes = data.slice(1).map((price, i) => price - data[i]);
  
  // Calculate initial average gain and loss
  let avgGain = 0;
  let avgLoss = 0;
  
  for (let i = 0; i < period; i++) {
    if (changes[i] >= 0) {
      avgGain += changes[i];
    } else {
      avgLoss -= changes[i];
    }
  }
  
  avgGain /= period;
  avgLoss /= period;
  
  // Calculate first RSI
  let rs = avgGain / avgLoss;
  let rsi = 100 - (100 / (1 + rs));
  
  const padding = new Array(period).fill(null);
  result.push(rsi);
  
  // Calculate remaining RSI values
  for (let i = period; i < changes.length; i++) {
    const change = changes[i];
    avgGain = ((avgGain * (period - 1)) + (change > 0 ? change : 0)) / period;
    avgLoss = ((avgLoss * (period - 1)) + (change < 0 ? -change : 0)) / period;
    
    rs = avgGain / avgLoss;
    rsi = 100 - (100 / (1 + rs));
    result.push(rsi);
  }
  
  return [...padding, ...result];
}

// Moving Average Convergence Divergence (MACD)
export interface MACDResult {
  macdLine: (number | null)[];
  signalLine: (number | null)[];
  histogram: (number | null)[];
}

export function calculateMACD(data: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9): MACDResult {
  const fastEMA = calculateEMA(data, fastPeriod);
  const slowEMA = calculateEMA(data, slowPeriod);
  
  // Calculate MACD line
  const macdLine = fastEMA.map((fast, i) => {
    if (fast === null || slowEMA[i] === null) return null;
    return fast - slowEMA[i];
  });
  
  // Calculate Signal line (EMA of MACD line)
  const signalLine = calculateEMA(macdLine.filter(x => x !== null) as number[], signalPeriod);
  
  // Calculate Histogram
  const histogram = macdLine.map((macd, i) => {
    if (macd === null || signalLine[i] === null) return null;
    return macd - signalLine[i];
  });
  
  return {
    macdLine,
    signalLine,
    histogram
  };
}

// Bollinger Bands
export interface BollingerBands {
  middle: number[];
  upper: number[];
  lower: number[];
}

export function calculateBollingerBands(data: number[], period: number = 20, multiplier: number = 2): BollingerBands {
  const sma = calculateSMA(data, period);
  const result: BollingerBands = {
    middle: sma,
    upper: [],
    lower: []
  };
  
  // Calculate standard deviation for each period
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const mean = sma[i];
    const squaredDiffs = slice.map(price => Math.pow(price - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / period;
    const stdDev = Math.sqrt(variance);
    
    result.upper.push(mean + (multiplier * stdDev));
    result.lower.push(mean - (multiplier * stdDev));
  }
  
  // Pad the beginning with nulls
  const padding = new Array(period - 1).fill(null);
  result.upper = [...padding, ...result.upper];
  result.lower = [...padding, ...result.lower];
  
  return result;
}

// Helper function to convert StockDataPoint array to price array
export function getPrices(data: StockDataPoint[]): number[] {
  return data.map(point => point.close);
} 