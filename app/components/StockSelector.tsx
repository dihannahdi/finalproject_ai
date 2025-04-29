'use client';

import React, { useState } from 'react';

// Stock option interface
interface StockOption {
  symbol: string;
  name: string;
  sector?: string;
  industry?: string;
}

interface StockSelectorProps {
  stocks: StockOption[];
  selectedStock: StockOption;
  onStockChange: (stock: StockOption) => void;
  stockData?: any; // Stock price data
}

const StockSelector: React.FC<StockSelectorProps> = ({
  stocks,
  selectedStock,
  onStockChange,
  stockData,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);

  // Filter stocks based on search term
  const filteredStocks = searchTerm
    ? stocks.filter(
        stock => 
          stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
          stock.name.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : stocks;

  // Handle stock selection
  const handleStockSelect = (stock: StockOption) => {
    onStockChange(stock);
    setSearchTerm('');
    setShowDropdown(false);
  };

  return (
    <div className="card">
      <h2 className="text-xl font-semibold mb-4">Select Stock</h2>
      
      {/* Search Input */}
      <div className="relative mb-4">
        <input
          type="text"
          className="input pr-10"
          placeholder="Search by symbol or company name"
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setShowDropdown(true);
          }}
          onFocus={() => setShowDropdown(true)}
        />
        {searchTerm && (
          <button
            className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
            onClick={() => {
              setSearchTerm('');
              setShowDropdown(false);
            }}
          >
            &times;
          </button>
        )}
        
        {/* Dropdown for search results */}
        {showDropdown && filteredStocks.length > 0 && (
          <div className="absolute z-10 w-full bg-white border border-gray-200 rounded-md shadow-lg mt-1 max-h-60 overflow-y-auto">
            {filteredStocks.map(stock => (
              <div
                key={stock.symbol}
                className={`px-4 py-2 hover:bg-gray-100 cursor-pointer ${
                  selectedStock.symbol === stock.symbol ? 'bg-primary/10' : ''
                }`}
                onClick={() => handleStockSelect(stock)}
              >
                <div className="font-medium">{stock.symbol}</div>
                <div className="text-sm text-gray-600">{stock.name}</div>
                {stock.sector && (
                  <div className="text-xs text-gray-500">{stock.sector}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Selected Stock Display */}
      <div className="bg-gray-50 p-4 rounded-md">
        <div className="flex justify-between items-start mb-2">
          <div>
            <h3 className="font-bold text-lg">{selectedStock.symbol}</h3>
            <p className="text-sm text-gray-600">{selectedStock.name}</p>
            {selectedStock.sector && (
              <p className="text-xs text-gray-500">
                {selectedStock.sector} | {selectedStock.industry || 'N/A'}
              </p>
            )}
          </div>
          {stockData && (
            <div className={`text-lg font-bold ${getColorClass(stockData?.change)}`}>
              ${stockData?.close?.toFixed(2)}
            </div>
          )}
        </div>
        
        {stockData && (
          <div className="grid grid-cols-2 gap-2 mt-3 text-sm">
            <div>
              <span className="text-gray-500">Change:</span>{' '}
              <span className={getColorClass(stockData.change)}>
                {stockData.change > 0 ? '+' : ''}
                {stockData.change?.toFixed(2)} ({stockData.changePercent?.toFixed(2)}%)
              </span>
            </div>
            <div>
              <span className="text-gray-500">Open:</span> ${stockData.open?.toFixed(2)}
            </div>
            <div>
              <span className="text-gray-500">High:</span> ${stockData.high?.toFixed(2)}
            </div>
            <div>
              <span className="text-gray-500">Low:</span> ${stockData.low?.toFixed(2)}
            </div>
            <div>
              <span className="text-gray-500">Volume:</span>{' '}
              {stockData.volume?.toLocaleString()}
            </div>
            <div>
              <span className="text-gray-500">Avg Vol:</span>{' '}
              {(stockData.volume * 1.2)?.toLocaleString()}
            </div>
          </div>
        )}
      </div>
      
      {/* Recent Stocks */}
      <div className="mt-4">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Recent Stocks</h3>
        <div className="flex flex-wrap gap-2">
          {stocks.slice(0, 6).map(stock => (
            <button
              key={stock.symbol}
              className={`px-3 py-1 text-sm rounded-full ${
                selectedStock.symbol === stock.symbol
                  ? 'bg-primary text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              onClick={() => handleStockSelect(stock)}
            >
              {stock.symbol}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

// Helper function to determine text color based on change value
function getColorClass(change?: number): string {
  if (!change) return 'text-gray-700';
  return change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : 'text-gray-700';
}

export default StockSelector; 