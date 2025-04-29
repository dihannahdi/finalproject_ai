'use client';

import React from 'react';

// Technical indicator interface
interface TechnicalIndicator {
  id: string;
  name: string;
  description?: string;
}

interface TechnicalIndicatorsProps {
  indicators: TechnicalIndicator[];
  selectedIndicators: string[];
  onIndicatorChange: (indicators: string[]) => void;
}

const TechnicalIndicators: React.FC<TechnicalIndicatorsProps> = ({
  indicators,
  selectedIndicators,
  onIndicatorChange,
}) => {
  const handleIndicatorToggle = (id: string) => {
    if (selectedIndicators.includes(id)) {
      onIndicatorChange(selectedIndicators.filter(i => i !== id));
    } else {
      onIndicatorChange([...selectedIndicators, id]);
    }
  };

  return (
    <div className="card">
      <h2 className="text-xl font-semibold mb-4">Technical Indicators</h2>
      
      <div className="space-y-3">
        {indicators.map(indicator => (
          <div key={indicator.id} className="flex items-center">
            <input
              type="checkbox"
              id={`indicator-${indicator.id}`}
              className="form-checkbox h-4 w-4 text-primary rounded"
              checked={selectedIndicators.includes(indicator.id)}
              onChange={() => handleIndicatorToggle(indicator.id)}
            />
            <label htmlFor={`indicator-${indicator.id}`} className="ml-2">
              <span className="text-sm font-medium text-gray-700">{indicator.name}</span>
              {indicator.description && (
                <p className="text-xs text-gray-500">{indicator.description}</p>
              )}
            </label>
          </div>
        ))}
      </div>
      
      {/* Indicator Information */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <h3 className="text-sm font-medium mb-2">About Technical Indicators</h3>
        <div className="text-xs text-gray-600">
          <p className="mb-2">Technical indicators are mathematical calculations based on price, volume, or open interest of a security. They help identify patterns and predict future price movements.</p>
          
          <div className="mt-3 space-y-2">
            {selectedIndicators.length > 0 ? (
              selectedIndicators.map(id => {
                const indicator = indicators.find(i => i.id === id);
                return indicator ? (
                  <div key={id} className="bg-gray-50 p-2 rounded">
                    <span className="font-medium">{indicator.name}:</span> {getIndicatorDescription(id)}
                  </div>
                ) : null;
              })
            ) : (
              <p className="italic">Select indicators to see descriptions</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper function to get detailed descriptions of indicators
function getIndicatorDescription(id: string): string {
  switch (id) {
    case 'sma':
      return 'Simple Moving Average calculates the average of a selected range of prices over a specific time period, helping to identify trends by smoothing out price fluctuations.';
    case 'ema':
      return 'Exponential Moving Average gives more weight to recent prices, making it more responsive to new information than the Simple Moving Average.';
    case 'rsi':
      return 'Relative Strength Index measures the speed and change of price movements, used to identify overbought or oversold conditions in the market.';
    case 'macd':
      return 'Moving Average Convergence Divergence shows the relationship between two moving averages of a security\'s price, helping to identify momentum and trend direction.';
    case 'bollinger':
      return 'Bollinger Bands consist of a middle SMA along with upper and lower bands that define price volatility, helping to identify potential overbought or oversold conditions.';
    case 'volume':
      return 'Volume indicators analyze the strength of price trends based on trading volumes, confirming price movements or identifying potential reversals.';
    case 'atr':
      return 'Average True Range measures market volatility by decomposing the entire range of a security\'s price for that period, useful for setting stop losses or entry points.';
    default:
      return 'A technical analysis tool that helps identify market trends and potential trading opportunities.';
  }
}

export default TechnicalIndicators; 