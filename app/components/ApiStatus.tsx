'use client';

import React, { useState, useEffect } from 'react';

interface ApiStatusProps {
  apiKey: string;
}

export default function ApiStatus({ apiKey }: ApiStatusProps) {
  const [status, setStatus] = useState<'demo' | 'active' | 'unknown'>('unknown');
  
  useEffect(() => {
    if (!apiKey) {
      setStatus('unknown');
    } else if (apiKey === 'demo') {
      setStatus('demo');
    } else {
      setStatus('active');
    }
  }, [apiKey]);
  
  return (
    <div className="flex items-center text-xs">
      <span 
        className={`inline-flex items-center px-2 py-1 rounded-full border ${
          status === 'active' 
            ? 'bg-green-100 text-green-800 border-green-200' 
            : status === 'demo' 
              ? 'bg-yellow-100 text-yellow-800 border-yellow-200'
              : 'bg-gray-100 text-gray-800 border-gray-200'
        }`}
      >
        <span className={`w-2 h-2 rounded-full mr-1 ${
          status === 'active' ? 'bg-green-500' : 
          status === 'demo' ? 'bg-yellow-500' : 'bg-gray-500'
        }`}></span>
        {status === 'active' && 'Alpha Vantage API Active'}
        {status === 'demo' && 'Demo API Key'}
        {status === 'unknown' && 'API Status Unknown'}
      </span>
      {status === 'demo' && (
        <span className="ml-2 text-gray-500">
          (Limited data available - <a href="https://www.alphavantage.co/support/#api-key" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Get free API key</a>)
        </span>
      )}
    </div>
  );
} 