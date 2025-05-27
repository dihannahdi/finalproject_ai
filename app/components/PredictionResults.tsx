'use client';

import React from 'react';
import { PredictionPoint } from '@/app/lib/api';

interface PredictionResultsProps {
  predictions: PredictionPoint[];
  algorithmName: string;
  stockSymbol: string;
  currentPrice?: number;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({
  predictions,
  algorithmName,
  stockSymbol,
  currentPrice,
}) => {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="card p-6">
        <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
        <p className="text-gray-500">No predictions available</p>
      </div>
    );
  }

  // Get the last prediction (30 days out)
  const finalPrediction = predictions[predictions.length - 1];
  
  // Check if we're using a fallback algorithm
  const isFallback = algorithmName.includes('fallback');
  
  // Calculate predicted change from current price
  const priceChange = currentPrice 
    ? finalPrediction.price - currentPrice
    : 0;
  
  const percentChange = currentPrice 
    ? (priceChange / currentPrice) * 100
    : 0;
  
  // Determine if prediction is bullish, bearish, or neutral
  const sentiment = percentChange > 5 
    ? 'Bullish' 
    : percentChange < -5 
      ? 'Bearish' 
      : 'Neutral';
  
  // Determine risk level based on volatility
  const volatility = predictions.reduce((total, pred) => {
    return total + (pred.upper - pred.lower) / pred.price;
  }, 0) / predictions.length;
  
  // More reasonable risk level calculation based on actual market conditions
  // Typical volatility for stable stocks is 1-2%, moderate is 2-4%, high is >4%
  const riskLevel = volatility > 0.04 
    ? 'High' 
    : volatility > 0.02 
      ? 'Medium' 
      : 'Low';
      
  // More reasonable confidence calculation based on prediction interval size
  // Financial models typically have 75-95% confidence depending on market conditions
  const confidenceLevel = Math.min(95, Math.max(75, 95 - (volatility * 250))).toFixed(1);

  return (
    <div className="card">
      <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
      
      {/* Algorithm Info */}
      <div className="mb-4">
        <div className="flex items-center">
          <span className="text-sm font-medium text-gray-700 mr-2">Algorithm:</span>
          <span className={`text-sm px-2 py-0.5 rounded ${
            algorithmName.includes('fallback') 
              ? 'bg-yellow-100' 
              : algorithmName.includes('(') && algorithmName.includes(')') 
                ? 'bg-orange-100'
                : 'bg-green-100'
          }`}>{algorithmName}</span>
        </div>
        
        {algorithmName.includes('fallback') && (
          <div className="mt-2 p-3 bg-yellow-50 border-l-4 border-yellow-400 text-sm text-yellow-800">
            <div className="font-medium">Using fallback algorithm</div>
            <p>The requested algorithm encountered issues and a fallback was used. Predictions may be less accurate than expected.</p>
          </div>
        )}
        
        {(algorithmName.includes('(') && algorithmName.includes(')') && !algorithmName.includes('fallback')) && (
          <div className="mt-2 p-3 bg-orange-50 border-l-4 border-orange-400 text-sm text-orange-800">
            <div className="font-medium">Using alternative implementation</div>
            <p>The algorithm is using a different model than requested. This may affect prediction accuracy.</p>
          </div>
        )}
      </div>
      
      <div className="mb-6">
        <h3 className="text-lg font-medium text-gray-700">Forecast Summary</h3>
        <p className="text-2xl font-bold text-primary mt-1">
          ${finalPrediction.price.toFixed(2)}
        </p>
        <p className="text-sm text-gray-600">
          Predicted price in 30 days using {algorithmName}
        </p>
        
        {currentPrice && (
          <div className="mt-2">
            <div className={`text-sm ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {priceChange >= 0 ? '↑' : '↓'} ${Math.abs(priceChange).toFixed(2)} ({percentChange.toFixed(2)}%)
            </div>
            <div className="text-sm mt-1 flex items-center">
              <span className="mr-2">Sentiment:</span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                sentiment === 'Bullish' 
                  ? 'bg-green-100 text-green-800' 
                  : sentiment === 'Bearish' 
                    ? 'bg-red-100 text-red-800' 
                    : 'bg-gray-100 text-gray-800'
              }`}>
                {sentiment}
              </span>
            </div>
          </div>
        )}
      </div>
      
      {/* Prediction Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">Confidence Level</p>
          <p className="text-xl font-bold text-primary">
            {confidenceLevel}%
          </p>
        </div>
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">Risk Level</p>
          <p className="text-xl font-bold text-primary flex items-center">
            {riskLevel}
            <span className={`ml-2 inline-block w-2.5 h-2.5 rounded-full ${
              riskLevel === 'High' 
                ? 'bg-red-500' 
                : riskLevel === 'Medium' 
                  ? 'bg-yellow-500' 
                  : 'bg-green-500'
            }`}></span>
          </p>
        </div>
      </div>
      
      {/* Prediction Table */}
      <div>
        <h3 className="text-md font-medium mb-2">Price Forecast</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 text-sm">Date</th>
                <th className="text-right py-2 text-sm">Price</th>
                <th className="text-right py-2 text-sm">Change</th>
              </tr>
            </thead>
            <tbody>
              {/* Every 5th prediction to keep the table manageable */}
              {predictions
                .filter((_, index) => index % 5 === 0 || index === predictions.length - 1)
                .map((prediction, i) => (
                <tr key={i} className="border-b border-gray-100">
                  <td className="py-2 text-sm">{prediction.date}</td>
                  <td className="text-right py-2 text-sm">
                    ${prediction.price.toFixed(2)}
                    <div className="text-xs text-gray-500">
                      ${prediction.lower.toFixed(2)} - ${prediction.upper.toFixed(2)}
                    </div>
                  </td>
                  <td className={`text-right py-2 text-sm ${
                    currentPrice && prediction.price > currentPrice 
                      ? 'text-green-600' 
                      : 'text-red-600'
                  }`}>
                    {currentPrice 
                      ? ((prediction.price / currentPrice - 1) * 100).toFixed(2) 
                      : '0.00'
                    }%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Prediction Notes */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <h3 className="text-md font-medium mb-2">Analysis Notes</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• {stockSymbol} is predicted to {priceChange >= 0 ? 'increase' : 'decrease'} by {Math.abs(percentChange).toFixed(2)}% over the next 30 days.</li>
          <li>• Predictions are based on historical price patterns and {algorithmName} algorithm.</li>
          <li>• The prediction confidence level is {confidenceLevel}%, indicating {riskLevel.toLowerCase()} volatility.</li>
          <li>• Results are for informational purposes only and not financial advice.</li>
        </ul>
      </div>
      
      <div className="mt-4 text-xs text-gray-500 italic">
        <p>Last updated: {new Date().toLocaleDateString()} {new Date().toLocaleTimeString()}</p>
      </div>
    </div>
  );
};

export default PredictionResults; 