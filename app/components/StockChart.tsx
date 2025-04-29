'use client';

import React, { useEffect, useRef } from 'react';
import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import dynamic from 'next/dynamic';

// Import ApexCharts dynamically to avoid SSR issues
const ReactApexChart = dynamic(() => import('react-apexcharts'), { ssr: false });

interface StockChartProps {
  historicalData: StockDataPoint[];
  predictions?: PredictionPoint[];
  symbol: string;
  algorithmName?: string;
  indicators?: string[];
  isLoading?: boolean;
}

const StockChart: React.FC<StockChartProps> = ({
  historicalData,
  predictions = [],
  symbol,
  algorithmName = '',
  indicators = [],
  isLoading = false,
}) => {
  // Skip rendering if the component is being server-side rendered
  if (typeof window === 'undefined') {
    return null;
  }

  if (isLoading) {
    return (
      <div className="h-96 flex items-center justify-center">
        <p className="text-gray-500">Loading chart data...</p>
      </div>
    );
  }

  if (!historicalData || historicalData.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center">
        <p className="text-gray-500">No data available</p>
      </div>
    );
  }

  // Prepare data for the chart
  const dates = historicalData.map(item => item.date);
  const prices = historicalData.map(item => item.close);
  
  // Prepare prediction data if available
  const predictionDates = predictions.map(item => item.date);
  const predictionPrices = predictions.map(item => item.price);
  const upperBounds = predictions.map(item => item.upper);
  const lowerBounds = predictions.map(item => item.lower);

  // Prepare technical indicators if requested
  const sma20 = indicators.includes('sma') ? calculateSMA(prices, 20) : [];
  const ema9 = indicators.includes('ema') ? calculateEMA(prices, 9) : [];
  
  // Combine historical and prediction dates
  const allDates = [...dates, ...predictionDates];
  
  // Chart options
  const options = {
    chart: {
      height: 380,
      type: 'line' as const,
      toolbar: {
        show: true,
        tools: {
          download: true,
          selection: true,
          zoom: true,
          zoomin: true,
          zoomout: true,
          pan: true,
        },
      },
      animations: {
        enabled: true,
      },
    },
    stroke: {
      width: [2, 2, 1, 1, 2, 2],
      curve: 'smooth' as const,
      dashArray: [0, 0, 0, 0, 0, 0],
    },
    colors: ['#0066cc', '#00cc99', '#f59e0b', '#f59e0b', '#6366f1', '#8b5cf6'],
    title: {
      text: `${symbol} Stock Price Chart`,
      align: 'left' as const,
    },
    subtitle: {
      text: algorithmName ? `With ${algorithmName} Predictions` : '',
      align: 'left' as const,
    },
    labels: allDates,
    xaxis: {
      type: 'datetime' as const,
      labels: {
        datetimeUTC: false,
      },
    },
    yaxis: {
      title: {
        text: 'Price',
      },
      labels: {
        formatter: (value: number) => `$${value.toFixed(2)}`,
      },
    },
    tooltip: {
      shared: true,
      intersect: false,
      y: {
        formatter: (value: number) => `$${value.toFixed(2)}`,
      },
    },
    legend: {
      position: 'top' as const,
      horizontalAlign: 'right' as const,
    },
    grid: {
      borderColor: '#f1f1f1',
    },
    annotations: {
      xaxis: [
        {
          x: new Date(dates[dates.length - 1]).getTime(),
          borderColor: '#999',
          label: {
            text: 'Prediction Start',
            style: {
              color: '#fff',
              background: '#999',
            },
          },
        },
      ],
    },
  };

  // Prepare series data
  const series = [
    {
      name: 'Historical Price',
      data: prices.map((price, index) => [new Date(dates[index]).getTime(), price]),
    },
  ];

  // Add prediction series if available
  if (predictions.length > 0) {
    series.push({
      name: 'Predicted Price',
      data: predictionPrices.map((price, index) => [new Date(predictionDates[index]).getTime(), price]),
    });

    // Add upper and lower bounds
    series.push({
      name: 'Upper Bound',
      data: upperBounds.map((price, index) => [new Date(predictionDates[index]).getTime(), price]),
    });
    
    series.push({
      name: 'Lower Bound',
      data: lowerBounds.map((price, index) => [new Date(predictionDates[index]).getTime(), price]),
    });
  }

  // Add technical indicators if requested
  if (indicators.includes('sma')) {
    series.push({
      name: 'SMA (20)',
      data: sma20.map((value, index) => [new Date(dates[index]).getTime(), value]),
    });
  }

  if (indicators.includes('ema')) {
    series.push({
      name: 'EMA (9)',
      data: ema9.map((value, index) => [new Date(dates[index]).getTime(), value]),
    });
  }

  return (
    <div className="stock-chart">
      <ReactApexChart 
        options={options} 
        series={series} 
        type="line" 
        height={380} 
      />
    </div>
  );
};

// Calculate Simple Moving Average
function calculateSMA(data: number[], window: number): number[] {
  const result: number[] = [];
  
  // Need at least 'window' data points
  if (data.length < window) {
    return new Array(data.length).fill(null);
  }
  
  // Calculate initial sum
  let sum = 0;
  for (let i = 0; i < window; i++) {
    sum += data[i];
  }
  
  // First SMA value
  result.push(sum / window);
  
  // Calculate remaining SMA values
  for (let i = window; i < data.length; i++) {
    sum = sum - data[i - window] + data[i];
    result.push(sum / window);
  }
  
  // Pad the beginning with nulls
  const padding = new Array(window - 1).fill(null);
  return [...padding, ...result];
}

// Calculate Exponential Moving Average
function calculateEMA(data: number[], window: number): number[] {
  const result: number[] = [];
  const k = 2 / (window + 1);
  
  // Need at least 'window' data points
  if (data.length < window) {
    return new Array(data.length).fill(null);
  }
  
  // Calculate SMA for the first EMA value
  let ema = data.slice(0, window).reduce((sum, price) => sum + price, 0) / window;
  
  // Pad the beginning with nulls
  const padding = new Array(window - 1).fill(null);
  result.push(ema);
  
  // Calculate EMA for remaining data points
  for (let i = window; i < data.length; i++) {
    ema = (data[i] - ema) * k + ema;
    result.push(ema);
  }
  
  return [...padding, ...result];
}

export default StockChart; 