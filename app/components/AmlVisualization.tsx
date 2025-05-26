'use client';

import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const AmlVisualization: React.FC = () => {
  // Sample data - In a real application, this would come from your backend
  const transactionData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Normal Transactions',
        data: [1200, 1900, 1500, 2100, 1800, 2400],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: 'Suspicious Transactions',
        data: [100, 150, 200, 180, 250, 300],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
    ],
  };

  const riskScoreData = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
    datasets: [
      {
        data: [65, 25, 10],
        backgroundColor: [
          'rgba(75, 192, 192, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(255, 99, 132, 0.6)',
        ],
        borderColor: [
          'rgb(75, 192, 192)',
          'rgb(255, 206, 86)',
          'rgb(255, 99, 132)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const alertTrendData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'AML Alerts',
        data: [12, 19, 15, 21, 18, 24],
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
    ],
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">AI/AML Analysis Dashboard</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Transaction Pattern Chart */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">Transaction Patterns</h3>
          <Line
            data={transactionData}
            options={{
              responsive: true,
              plugins: {
                legend: {
                  position: 'top' as const,
                },
                title: {
                  display: true,
                  text: 'Monthly Transaction Volume',
                },
              },
            }}
          />
        </div>

        {/* Risk Score Distribution */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">Risk Score Distribution</h3>
          <Doughnut
            data={riskScoreData}
            options={{
              responsive: true,
              plugins: {
                legend: {
                  position: 'top' as const,
                },
                title: {
                  display: true,
                  text: 'Customer Risk Distribution',
                },
              },
            }}
          />
        </div>

        {/* Alert Trends */}
        <div className="p-4 bg-gray-50 rounded-lg md:col-span-2">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">Alert Trends</h3>
          <Bar
            data={alertTrendData}
            options={{
              responsive: true,
              plugins: {
                legend: {
                  position: 'top' as const,
                },
                title: {
                  display: true,
                  text: 'Monthly AML Alert Volume',
                },
              },
            }}
          />
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="text-sm font-semibold text-blue-700">Total Transactions</h4>
          <p className="text-2xl font-bold text-blue-900">10,890</p>
        </div>
        <div className="p-4 bg-yellow-50 rounded-lg">
          <h4 className="text-sm font-semibold text-yellow-700">Active Alerts</h4>
          <p className="text-2xl font-bold text-yellow-900">24</p>
        </div>
        <div className="p-4 bg-green-50 rounded-lg">
          <h4 className="text-sm font-semibold text-green-700">Risk Score</h4>
          <p className="text-2xl font-bold text-green-900">Low</p>
        </div>
      </div>
    </div>
  );
};

export default AmlVisualization; 