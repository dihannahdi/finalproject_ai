'use client';

import React from 'react';

interface AlgorithmStatusProps {
  requestedAlgorithm: string;
  actualAlgorithm: string | null;
}

const AlgorithmStatus: React.FC<AlgorithmStatusProps> = ({
  requestedAlgorithm,
  actualAlgorithm,
}) => {
  // If we don't have actual algorithm info yet
  if (!actualAlgorithm) {
    return (
      <div className="flex items-center">
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
          <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Requested {requestedAlgorithm}
        </span>
      </div>
    );
  }

  // If the actual algorithm matches the requested one exactly
  if (actualAlgorithm === requestedAlgorithm) {
    return (
      <div className="flex items-center">
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
          <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          Using {requestedAlgorithm}
        </span>
      </div>
    );
  }

  // Check different fallback scenarios
  const isFallback = actualAlgorithm.includes('fallback');
  const isPartialFallback = actualAlgorithm.includes('(') && actualAlgorithm.includes(')');
  
  return (
    <div className="flex flex-col">
      <div className="flex items-center">
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
          isFallback 
            ? 'bg-yellow-100 text-yellow-800' 
            : isPartialFallback 
              ? 'bg-orange-100 text-orange-800'
              : 'bg-blue-100 text-blue-800'
        }`}>
          {isFallback ? (
            <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          ) : (
            <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
          Using {actualAlgorithm}
        </span>
      </div>
      
      {/* Show requested algorithm if different from actual */}
      {requestedAlgorithm.toLowerCase() !== actualAlgorithm.toLowerCase() && (
        <div className="mt-1 text-xs text-gray-500 flex items-center">
          <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
          </svg>
          Requested: {requestedAlgorithm}
        </div>
      )}
      
      <p className="text-xs text-gray-500 mt-1">
        {isFallback 
          ? "Fallback algorithm used due to issues with the requested algorithm."
          : isPartialFallback
            ? "This algorithm is using a different model as a fallback."
            : "Alternative algorithm used for better results."}
      </p>
    </div>
  );
};

export default AlgorithmStatus; 