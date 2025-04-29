'use client';

import React, { useState } from 'react';
import { getDateRangeForLastDays, getDateRangeForLastMonths, getDateRangeForLastYears } from '@/app/hooks/useStockData';

interface DateRangePickerProps {
  startDate: string;
  endDate: string;
  onChange: (range: { start: string; end: string }) => void;
}

const DateRangePicker: React.FC<DateRangePickerProps> = ({
  startDate,
  endDate,
  onChange,
}) => {
  const [customRange, setCustomRange] = useState(false);

  const handleQuickRangeSelect = (option: string) => {
    let range;
    
    switch (option) {
      case '1w':
        range = getDateRangeForLastDays(7);
        break;
      case '1m':
        range = getDateRangeForLastMonths(1);
        break;
      case '3m':
        range = getDateRangeForLastMonths(3);
        break;
      case '6m':
        range = getDateRangeForLastMonths(6);
        break;
      case '1y':
        range = getDateRangeForLastYears(1);
        break;
      case '5y':
        range = getDateRangeForLastYears(5);
        break;
      default:
        return;
    }
    
    setCustomRange(false);
    onChange(range);
  };

  const handleCustomDateChange = (field: 'start' | 'end', value: string) => {
    onChange({
      start: field === 'start' ? value : startDate,
      end: field === 'end' ? value : endDate,
    });
  };

  return (
    <div className="card">
      <h2 className="text-xl font-semibold mb-4">Date Range</h2>
      
      {/* Quick selection buttons */}
      <div className="flex flex-wrap gap-2 mb-4">
        <button
          className="px-3 py-1 text-sm rounded-full bg-gray-200 text-gray-700 hover:bg-gray-300"
          onClick={() => handleQuickRangeSelect('1w')}
        >
          1W
        </button>
        <button
          className="px-3 py-1 text-sm rounded-full bg-gray-200 text-gray-700 hover:bg-gray-300"
          onClick={() => handleQuickRangeSelect('1m')}
        >
          1M
        </button>
        <button
          className="px-3 py-1 text-sm rounded-full bg-gray-200 text-gray-700 hover:bg-gray-300"
          onClick={() => handleQuickRangeSelect('3m')}
        >
          3M
        </button>
        <button
          className="px-3 py-1 text-sm rounded-full bg-gray-200 text-gray-700 hover:bg-gray-300"
          onClick={() => handleQuickRangeSelect('6m')}
        >
          6M
        </button>
        <button
          className="px-3 py-1 text-sm rounded-full bg-gray-200 text-gray-700 hover:bg-gray-300"
          onClick={() => handleQuickRangeSelect('1y')}
        >
          1Y
        </button>
        <button
          className="px-3 py-1 text-sm rounded-full bg-gray-200 text-gray-700 hover:bg-gray-300"
          onClick={() => handleQuickRangeSelect('5y')}
        >
          5Y
        </button>
        <button
          className={`px-3 py-1 text-sm rounded-full ${
            customRange
              ? 'bg-primary text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => setCustomRange(true)}
        >
          Custom
        </button>
      </div>
      
      {/* Custom date range */}
      {customRange && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Start Date</label>
            <input
              type="date"
              className="input"
              value={startDate}
              onChange={(e) => handleCustomDateChange('start', e.target.value)}
            />
          </div>
          <div>
            <label className="label">End Date</label>
            <input
              type="date"
              className="input"
              value={endDate}
              onChange={(e) => handleCustomDateChange('end', e.target.value)}
            />
          </div>
        </div>
      )}
      
      {/* Date range display */}
      {!customRange && (
        <div className="text-sm text-gray-600 mt-2">
          <div>From: {formatDate(startDate)}</div>
          <div>To: {formatDate(endDate)}</div>
          <div className="mt-1">
            Total: {calculateDateDifference(startDate, endDate)} days
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function to format date for display
function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

// Helper function to calculate the difference between two dates in days
function calculateDateDifference(startDate: string, endDate: string): number {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const diffTime = Math.abs(end.getTime() - start.getTime());
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return diffDays;
}

export default DateRangePicker; 