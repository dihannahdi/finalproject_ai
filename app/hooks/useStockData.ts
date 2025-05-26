'use client';

import { useState, useEffect } from 'react';
import { fetchStockData, fetchRealTimeStockData, predictStockPrice } from '@/app/lib/api';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';

interface UseStockDataProps {
  symbol: string;
  startDate: string;
  endDate: string;
  algorithmId?: string;
  predictionDays?: number;
}

interface UseStockDataResult {
  historicalData: StockDataPoint[];
  predictions: PredictionPoint[];
  realtimeData: any | null;
  isLoading: boolean;
  error: string | null;
  refreshData: () => Promise<void>;
}

/**
 * Custom hook for fetching and managing stock data
 */
export default function useStockData({
  symbol,
  startDate,
  endDate,
  algorithmId = 'lstm',
  predictionDays = 30,
}: UseStockDataProps): UseStockDataResult {
  const [historicalData, setHistoricalData] = useState<StockDataPoint[]>([]);
  const [predictions, setPredictions] = useState<PredictionPoint[]>([]);
  const [realtimeData, setRealtimeData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Function to fetch all data
  const fetchData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Fetch historical data
      const stockData = await fetchStockData(symbol, startDate, endDate);
      if (!stockData || stockData.length === 0) {
        throw new Error('No historical data available for the selected period');
      }
      setHistoricalData(stockData);

      // Fetch real-time quote
      try {
        const realtime = await fetchRealTimeStockData(symbol);
        if (realtime && realtime.price) {
          setRealtimeData(realtime);
        } else {
          console.warn('Real-time data not available, using latest historical data point');
          const latestData = stockData[stockData.length - 1];
          setRealtimeData({
            symbol,
            price: latestData.close,
            change: latestData.close - stockData[stockData.length - 2].close,
            changePercent: ((latestData.close - stockData[stockData.length - 2].close) / stockData[stockData.length - 2].close * 100),
            volume: latestData.volume
          });
        }
      } catch (realtimeError) {
        console.warn('Error fetching real-time data:', realtimeError);
        // Fallback to latest historical data
        const latestData = stockData[stockData.length - 1];
        setRealtimeData({
          symbol,
          price: latestData.close,
          change: latestData.close - stockData[stockData.length - 2].close,
          changePercent: ((latestData.close - stockData[stockData.length - 2].close) / stockData[stockData.length - 2].close * 100),
          volume: latestData.volume
        });
      }

      // Generate predictions
      if (stockData.length > 0) {
        const predictionResults = await predictStockPrice(
          symbol,
          stockData,
          algorithmId,
          predictionDays
        );
        setPredictions(predictionResults);
      }

      setIsLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stock data');
      setIsLoading(false);
    }
  };

  // Fetch data when dependencies change
  useEffect(() => {
    fetchData();
  }, [symbol, startDate, endDate, algorithmId, predictionDays]);

  // Function to manually refresh data
  const refreshData = async () => {
    await fetchData();
  };

  return {
    historicalData,
    predictions,
    realtimeData,
    isLoading,
    error,
    refreshData,
  };
}

/**
 * Format to convert a Date object to YYYY-MM-DD string
 */
export function formatDateToYYYYMMDD(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

/**
 * Helper to get date range for the last N days
 */
export function getDateRangeForLastDays(days: number): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - days);
  
  return {
    start: formatDateToYYYYMMDD(start),
    end: formatDateToYYYYMMDD(end),
  };
}

/**
 * Helper to get date range for the last N months
 */
export function getDateRangeForLastMonths(months: number): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setMonth(end.getMonth() - months);
  
  return {
    start: formatDateToYYYYMMDD(start),
    end: formatDateToYYYYMMDD(end),
  };
}

/**
 * Helper to get date range for the last N years
 */
export function getDateRangeForLastYears(years: number): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - years);
  
  return {
    start: formatDateToYYYYMMDD(start),
    end: formatDateToYYYYMMDD(end),
  };
} 