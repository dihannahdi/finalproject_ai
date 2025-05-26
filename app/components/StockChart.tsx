'use client';

import React, { useEffect, useState } from 'react';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import dynamic from 'next/dynamic';
import { 
  calculateSMA, 
  calculateEMA, 
  calculateRSI, 
  calculateMACD, 
  calculateBollingerBands,
  getPrices,
  type MACDResult,
  type BollingerBands
} from '@/app/lib/technicalIndicators';
import type { ApexOptions } from 'apexcharts';

// Import ApexCharts dynamically to avoid SSR issues
const ReactApexChart = dynamic(() => import('react-apexcharts'), { 
  ssr: false,
  loading: () => (
    <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
      <p className="text-gray-500">Loading chart...</p>
    </div>
  )
});

interface StockChartProps {
  historicalData: StockDataPoint[];
  predictions?: PredictionPoint[];
  symbol: string;
  algorithmName?: string;
  indicators?: string[];
  isLoading?: boolean;
}

// Define series type
type SeriesData = {
  name: string;
  data: Array<{
    x: number;
    y: number | null;
  }>;
  type?: 'line' | 'column';
  yAxisIndex?: number;
};

const StockChart: React.FC<StockChartProps> = ({
  historicalData,
  predictions = [],
  symbol,
  algorithmName = '',
  indicators = [],
  isLoading = false,
}) => {
  const [isClient, setIsClient] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Add debug logging
  console.log('StockChart render:', {
    isClient,
    hasHistoricalData: !!historicalData,
    historicalDataLength: historicalData?.length,
    hasPredictions: !!predictions,
    predictionsLength: predictions?.length,
    predictionsData: predictions?.slice(0, 3), // Log first few predictions
    symbol,
    algorithmName,
    indicators
  });

  if (!isClient || isLoading || !historicalData || historicalData.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
        <p className="text-gray-500">{isLoading ? 'Loading chart data...' : 'No data available'}</p>
      </div>
    );
  }

  // Error handling for chart rendering
  if (errorMessage) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
        <div className="text-center p-4">
          <p className="text-red-500 font-medium">Chart Error</p>
          <p className="text-gray-600 text-sm mt-2">{errorMessage}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }

  try {
    // Prepare data for the chart
    const dates = historicalData.map(item => item.date);
    const prices = getPrices(historicalData);
    
    // Calculate technical indicators
    const sma20 = indicators.includes('sma') ? calculateSMA(prices, 20) : [];
    const ema9 = indicators.includes('ema') ? calculateEMA(prices, 9) : [];
    const rsi = indicators.includes('rsi') ? calculateRSI(prices, 14) : [];
    const macd = indicators.includes('macd') ? calculateMACD(prices) : null;
    const bollinger = indicators.includes('bollinger') ? calculateBollingerBands(prices) : null;
  
    // Prepare prediction data if available and ensure all values are valid numbers
    const predictionDates = predictions?.length ? predictions.map(item => item.date) : [];
    
    // Add additional validation for chart data
    const sanitizeValue = (value: any): number | null => {
      return (value !== null && value !== undefined && isFinite(value)) ? value : null;
    };
    
    const predictionPrices = predictions?.length 
      ? predictions.map(item => sanitizeValue(item.price))
      : [];
      
    const upperBounds = predictions?.length 
      ? predictions.map(item => sanitizeValue(item.upper))
      : [];
      
    const lowerBounds = predictions?.length 
      ? predictions.map(item => sanitizeValue(item.lower))
      : [];

    // Chart options
    const options: any = {
      chart: {
        type: 'line' as const,
        height: 380,
        animations: {
          enabled: false
        },
        toolbar: {
          show: true,
          tools: {
            download: true,
            selection: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true
          }
        }
      },
      title: {
        text: `${symbol} Stock Price${algorithmName ? ` - ${algorithmName}` : ''}`,
        align: 'left' as const
      },
      xaxis: {
        type: 'datetime' as const,
        categories: dates.map(date => new Date(date).getTime()),
        labels: {
          datetimeUTC: false
        }
      },
      yaxis: [
        {
          title: {
            text: 'Price'
          },
          labels: {
            formatter: (value: number) => value.toFixed(2)
          },
          logarithmic: false
        }
      ],
      stroke: {
        width: [2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        curve: 'smooth' as const,
        dashArray: [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0]
      },
      colors: [
        '#2563eb', // Price
        '#16a34a', // Predictions - Green
        '#dc2626', // Upper bound
        '#dc2626', // Lower bound
        '#9333ea', // SMA
        '#ea580c', // EMA
        '#0891b2', // RSI
        '#2563eb', // MACD line
        '#16a34a', // Signal line
        '#dc2626', // Histogram
        '#9333ea', // Bollinger Bands
      ],
      fill: {
        type: ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'gradient'],
        opacity: [1, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 0.1]
      },
      tooltip: {
        shared: true,
        intersect: false,
        x: {
          format: 'MMM dd, yyyy'
        }
      },
      legend: {
        position: 'top' as const,
        horizontalAlign: 'right' as const
      },
      markers: {
        size: [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        shape: 'circle' as const
      }
    };

    // Add additional y-axes for technical indicators if needed
    if (indicators.includes('rsi')) {
      options.yaxis.push({
        opposite: true,
        title: {
          text: 'RSI'
        },
        min: 0,
        max: 100,
        labels: {
          formatter: (value: number) => value.toFixed(0)
        },
        logarithmic: false
      });
    }

    if (indicators.includes('macd')) {
      options.yaxis.push({
        opposite: true,
        title: {
          text: 'MACD'
        },
        labels: {
          formatter: (value: number) => value.toFixed(2)
        },
        logarithmic: false
      });
    }

    // Update series type
    const series: SeriesData[] = [
      {
        name: 'Price',
        data: prices.map((price, index) => ({
          x: new Date(dates[index]).getTime(),
          y: sanitizeValue(price)
        })),
        type: 'line'
      }
    ];

    // Add prediction series if available
    if (predictions && predictions.length > 0) {
      // Debug logging for predictions
      console.log('Adding prediction series:', {
        predictionDates: predictionDates.slice(0, 3), // Show first few dates
        predictionPrices: predictionPrices.slice(0, 3), // Show first few prices
        upperBounds: upperBounds.slice(0, 3), // Show first few upper bounds
        lowerBounds: lowerBounds.slice(0, 3), // Show first few lower bounds
        validPredictions: predictionPrices.filter(p => p !== null).length
      });

      // Validate that we have at least one valid prediction
      if (predictionPrices.some(p => p !== null)) {
        // Create a smooth connection between historical data and predictions
        // by including the last historical data point in the prediction series
        if (dates.length > 0 && predictionDates.length > 0) {
          const lastHistoricalDate = new Date(dates[dates.length - 1]).getTime();
          const lastHistoricalPrice = prices[prices.length - 1];
          
          // Add predicted price line with stronger visibility
          series.push({
            name: 'Predicted Price',
            data: [
              // Include the last historical point as the first point in prediction series
              {
                x: lastHistoricalDate,
                y: lastHistoricalPrice
              },
              // Then add all prediction points
              ...predictionPrices.map((price, index) => ({
                x: new Date(predictionDates[index]).getTime(),
                y: sanitizeValue(price)
              }))
            ],
            type: 'line'
          });

          // Add upper bound
          series.push({
            name: 'Upper Bound',
            data: [
              // Start upper bound from last historical point
              {
                x: lastHistoricalDate,
                y: lastHistoricalPrice * 1.01 // Small initial bound
              },
              // Then add all upper bound points
              ...upperBounds.map((price, index) => ({
                x: new Date(predictionDates[index]).getTime(),
                y: sanitizeValue(price)
              }))
            ],
            type: 'line'
          });

          // Add lower bound
          series.push({
            name: 'Lower Bound',
            data: [
              // Start lower bound from last historical point
              {
                x: lastHistoricalDate,
                y: lastHistoricalPrice * 0.99 // Small initial bound
              },
              // Then add all lower bound points
              ...lowerBounds.map((price, index) => ({
                x: new Date(predictionDates[index]).getTime(),
                y: sanitizeValue(price)
              }))
            ],
            type: 'line'
          });
        } else {
          // Fallback to original implementation if we can't create a smooth connection
          series.push({
            name: 'Predicted Price',
            data: predictionPrices.map((price, index) => ({
              x: new Date(predictionDates[index]).getTime(),
              y: sanitizeValue(price)
            })),
            type: 'line'
          });

          // Add upper bound
          series.push({
            name: 'Upper Bound',
            data: upperBounds.map((price, index) => ({
              x: new Date(predictionDates[index]).getTime(),
              y: sanitizeValue(price)
            })),
            type: 'line'
          });

          // Add lower bound
          series.push({
            name: 'Lower Bound',
            data: lowerBounds.map((price, index) => ({
              x: new Date(predictionDates[index]).getTime(),
              y: sanitizeValue(price)
            })),
            type: 'line'
          });
        }
      } else {
        console.warn('No valid prediction values found in the predictions array');
      }
    } else {
      console.warn('No prediction data available to display');
    }

    // Add technical indicators
    if (indicators.includes('sma')) {
      series.push({
        name: 'SMA (20)',
        data: sma20.map((value, index) => ({
          x: new Date(dates[index]).getTime(),
          y: value
        })),
        type: 'line'
      });
    }

    if (indicators.includes('ema')) {
      series.push({
        name: 'EMA (9)',
        data: ema9.map((value, index) => ({
          x: new Date(dates[index]).getTime(),
          y: value
        })),
        type: 'line'
      });
    }

    if (indicators.includes('rsi')) {
      series.push({
        name: 'RSI (14)',
        data: rsi.map((value, index) => ({
          x: new Date(dates[index]).getTime(),
          y: value
        })),
        type: 'line',
        yAxisIndex: 1
      });
    }

    if (indicators.includes('macd') && macd) {
      series.push(
        {
          name: 'MACD Line',
          data: macd.macdLine.map((value, index) => ({
            x: new Date(dates[index]).getTime(),
            y: value
          })),
          type: 'line',
          yAxisIndex: 2
        },
        {
          name: 'Signal Line',
          data: macd.signalLine.map((value, index) => ({
            x: new Date(dates[index]).getTime(),
            y: value
          })),
          type: 'line',
          yAxisIndex: 2
        },
        {
          name: 'MACD Histogram',
          data: macd.histogram.map((value, index) => ({
            x: new Date(dates[index]).getTime(),
            y: value
          })),
          type: 'column',
          yAxisIndex: 2
        }
      );
    }

    if (indicators.includes('bollinger') && bollinger) {
      series.push(
        {
          name: 'Bollinger Middle',
          data: bollinger.middle.map((value, index) => ({
            x: new Date(dates[index]).getTime(),
            y: value
          })),
          type: 'line'
        },
        {
          name: 'Bollinger Upper',
          data: bollinger.upper.map((value, index) => ({
            x: new Date(dates[index]).getTime(),
            y: value
          })),
          type: 'line'
        },
        {
          name: 'Bollinger Lower',
          data: bollinger.lower.map((value, index) => ({
            x: new Date(dates[index]).getTime(),
            y: value
          })),
          type: 'line'
        }
      );
    }

    return (
      <div className="stock-chart bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
        {typeof window !== 'undefined' && (
          <ReactApexChart 
            options={options} 
            series={series} 
            type="line" 
            height={380}
            width="100%"
          />
        )}
      </div>
    );
  } catch (error) {
    console.error('Error in StockChart:', error);
    setErrorMessage('An error occurred while rendering the chart.');
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
        <div className="text-center p-4">
          <p className="text-red-500 font-medium">Chart Error</p>
          <p className="text-gray-600 text-sm mt-2">An error occurred while rendering the chart.</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }
};

export default StockChart; 