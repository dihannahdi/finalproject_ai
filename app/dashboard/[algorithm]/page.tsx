'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Dashboard from '../page';

// This is a wrapper component that reads the algorithm from the URL
// and then renders the Dashboard with that algorithm pre-selected
export default function AlgorithmSpecificDashboard({ 
  params 
}: { 
  params: { algorithm: string } 
}) {
  const router = useRouter();
  const { algorithm } = params;

  // Validate algorithm and redirect to main dashboard if invalid
  useEffect(() => {
    // List of valid algorithms (must match the IDs in ALGORITHM_OPTIONS)
    const validAlgorithms = [
      'lstm', 'transformer', 'cnnlstm', 'gan', 'xgboost', 'randomforest', 'gradientboost',
      'ensemble', 'tddm', 'arima', 'prophet', 'movingaverage'
    ];

    // Redirect to main dashboard if algorithm is invalid
    if (!validAlgorithms.includes(algorithm)) {
      console.warn(`Invalid algorithm: ${algorithm}. Redirecting to main dashboard.`);
      router.push('/dashboard');
    }
  }, [algorithm, router]);

  // Pass the params and searchParams to the Dashboard component
  return <Dashboard params={{ initialAlgorithm: algorithm }} searchParams={{}} />;
} 