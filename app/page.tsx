'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-6 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-primary">StockPred Master</h1>
          <nav>
            <ul className="flex space-x-6">
              <li><Link href="/" className="text-gray-700 hover:text-primary">Home</Link></li>
              <li><Link href="/dashboard" className="text-gray-700 hover:text-primary">Dashboard</Link></li>
              <li><Link href="/algorithms" className="text-gray-700 hover:text-primary">Algorithms</Link></li>
              <li><Link href="/about" className="text-gray-700 hover:text-primary">About</Link></li>
            </ul>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 bg-gradient-to-r from-primary/10 to-secondary/10">
        <div className="container mx-auto px-4 flex flex-col md:flex-row items-center">
          <div className="md:w-1/2 mb-10 md:mb-0">
            <h2 className="text-4xl font-bold mb-6">Intelligent Stock Market Prediction</h2>
            <p className="text-lg mb-8">
              StockPred Master leverages cutting-edge algorithms to provide real-time stock price predictions
              and visualization features, helping you make informed investment decisions.
            </p>
            <div className="flex space-x-4">
              <Link 
                href="/dashboard" 
                className="btn btn-primary"
              >
                Get Started
              </Link>
              <Link 
                href="/about" 
                className="btn bg-white text-primary border border-primary hover:bg-gray-50"
              >
                Learn More
              </Link>
            </div>
          </div>
          <div className="md:w-1/2 flex justify-center">
            <div className="relative w-full max-w-lg h-80">
              {/* Placeholder for stock chart image */}
              <div className="w-full h-full bg-white rounded-lg shadow-lg flex items-center justify-center">
                <p className="text-xl text-gray-400">Stock Chart Visualization</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="card">
              <div className="h-14 w-14 rounded-full bg-primary/20 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-2">Real-Time Data Integration</h3>
              <p className="text-gray-600">
                Connect to financial data APIs to fetch real-time stock market data, including stock prices,
                trading volumes, and more.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="card">
              <div className="h-14 w-14 rounded-full bg-primary/20 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-2">Advanced Prediction Algorithms</h3>
              <p className="text-gray-600">
                Utilize state-of-the-art algorithms like LSTM, Transformers, CNN-LSTM hybrids, GANs, and more
                for accurate stock price predictions.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="card">
              <div className="h-14 w-14 rounded-full bg-primary/20 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-2">Interactive Visualizations</h3>
              <p className="text-gray-600">
                View historical and predicted stock price trends using interactive charts, technical indicators,
                and comprehensive visual analytics.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-primary">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-white mb-6">Ready to Predict the Market?</h2>
          <p className="text-xl text-white/90 mb-8 max-w-3xl mx-auto">
            Start using StockPred Master today and gain valuable insights for your investment decisions.
          </p>
          <Link 
            href="/dashboard" 
            className="btn bg-white text-primary hover:bg-gray-100 px-8 py-3 text-lg"
          >
            Get Started Now
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-12">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <h3 className="text-xl font-bold mb-4">StockPred Master</h3>
              <p className="text-gray-400">
                An intelligent stock market prediction app that leverages cutting-edge algorithms.
              </p>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Features</h4>
              <ul className="space-y-2">
                <li><Link href="/dashboard" className="text-gray-400 hover:text-white">Dashboard</Link></li>
                <li><Link href="/algorithms" className="text-gray-400 hover:text-white">Algorithms</Link></li>
                <li><Link href="/visualization" className="text-gray-400 hover:text-white">Visualization</Link></li>
                <li><Link href="/alerts" className="text-gray-400 hover:text-white">Real-Time Alerts</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Resources</h4>
              <ul className="space-y-2">
                <li><Link href="/docs" className="text-gray-400 hover:text-white">Documentation</Link></li>
                <li><Link href="/api" className="text-gray-400 hover:text-white">API</Link></li>
                <li><Link href="/faq" className="text-gray-400 hover:text-white">FAQ</Link></li>
                <li><Link href="/support" className="text-gray-400 hover:text-white">Support</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Contact</h4>
              <ul className="space-y-2">
                <li className="text-gray-400">info@stockpredmaster.com</li>
                <li className="text-gray-400">+1 (555) 123-4567</li>
                <li className="text-gray-400">123 Trading Ave, Financial District</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-700 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; {new Date().getFullYear()} StockPred Master. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
} 