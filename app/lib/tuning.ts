/**
 * Utility functions for hyperparameter tuning
 */

import { 
  StockDataPoint, 
  PredictionPoint,
  safeGetStockData,
  safePredictStockPrice 
} from '@/app/lib/api';
import { generateTuningConfigurations } from '@/app/models/config';
import { createModel } from '@/app/models';

/**
 * Interface for model evaluation metrics
 */
export interface EvaluationMetrics {
  loss: number;            // Mean absolute error between predictions and actual values
  confidence: number;      // Confidence score (0-100)
  consistencyScore: number; // Measures prediction consistency
  evaluationTime: number;  // Time taken to train and predict in milliseconds
  predictionStability: number; // Standard deviation of day-to-day changes
  algorithmName: string;   // Name of the algorithm used
  parameterName: string;   // Name of the parameter being tuned
  parameterValue: any;     // Value of the parameter being tuned
}

/**
 * Run a complete tuning experiment for a given algorithm and parameter
 * 
 * @param algorithmType - Type of algorithm to tune (e.g., 'lstm', 'xgboost')
 * @param paramToTune - Parameter to tune (e.g., 'learningRate', 'numTrees')
 * @param paramValues - Array of values to test for the parameter
 * @param symbol - Stock symbol to use for testing
 * @param startDate - Start date for historical data
 * @param endDate - End date for historical data
 * @returns Promise resolving to array of evaluation metrics for each parameter value
 */
export async function runTuningExperiment(
  algorithmType: string,
  paramToTune: string,
  paramValues: any[],
  symbol: string = 'AAPL',
  startDate: string = getDefaultStartDate(),
  endDate: string = getDefaultEndDate()
): Promise<EvaluationMetrics[]> {
  // Generate configurations for each parameter value
  const configs = generateTuningConfigurations(algorithmType, paramToTune, paramValues);
  
  // Fetch historical data once to use for all experiments
  console.log(`Fetching historical data for ${symbol} from ${startDate} to ${endDate}`);
  const historicalData = await safeGetStockData(symbol, startDate, endDate);
  
  if (!historicalData || historicalData.length < 100) {
    console.error('Insufficient historical data for tuning');
    return [];
  }
  
  // Split data into training and testing sets (80/20 split)
  const splitIndex = Math.floor(historicalData.length * 0.8);
  const trainingData = historicalData.slice(0, splitIndex);
  const testingData = historicalData.slice(splitIndex);
  
  console.log(`Data split: ${trainingData.length} training points, ${testingData.length} testing points`);
  
  // Evaluate each configuration
  const results: EvaluationMetrics[] = [];
  
  for (const config of configs) {
    try {
      console.log(`Evaluating ${algorithmType} with ${paramToTune} = ${config.paramValue}`);
      const startTime = performance.now();
      
      // Create model with the current configuration
      const model = createModel(algorithmType, config.config);
      
      // Train model
      await model.train(trainingData);
      
      // Generate predictions
      const predictions = await model.predict(trainingData, 30);
      
      // Calculate evaluation metrics
      const metrics = evaluatePredictions(
        predictions, 
        testingData, 
        algorithmType,
        paramToTune,
        config.paramValue,
        performance.now() - startTime
      );
      
      results.push(metrics);
      console.log(`Evaluation complete: loss = ${metrics.loss.toFixed(2)}, confidence = ${metrics.confidence.toFixed(2)}%`);
      
    } catch (error) {
      console.error(`Error evaluating configuration ${paramToTune} = ${config.paramValue}:`, error);
      // Add failed result with null metrics
      results.push({
        loss: 999,
        confidence: 0,
        consistencyScore: 0,
        evaluationTime: 0,
        predictionStability: 0,
        algorithmName: algorithmType,
        parameterName: paramToTune,
        parameterValue: config.paramValue
      });
    }
  }
  
  // Sort results by loss (lower is better)
  results.sort((a, b) => a.loss - b.loss);
  
  // Log the best configuration
  const bestConfig = results[0];
  console.log(`Best configuration: ${paramToTune} = ${bestConfig.parameterValue}`);
  console.log(`Metrics: loss = ${bestConfig.loss.toFixed(2)}, confidence = ${bestConfig.confidence.toFixed(2)}%`);
  
  return results;
}

/**
 * Evaluate prediction quality against actual data
 */
function evaluatePredictions(
  predictions: PredictionPoint[],
  actualData: StockDataPoint[],
  algorithmName: string,
  paramName: string,
  paramValue: any,
  evaluationTime: number
): EvaluationMetrics {
  // If predictions or actual data are insufficient, return default metrics
  if (!predictions || predictions.length === 0 || !actualData || actualData.length < 5) {
    return {
      loss: 999,
      confidence: 0,
      consistencyScore: 0,
      evaluationTime,
      predictionStability: 0,
      algorithmName,
      parameterName: paramName,
      parameterValue: paramValue
    };
  }
  
  // Calculate mean absolute error (loss)
  let totalError = 0;
  let availableComparisons = 0;
  
  // We can only compare predictions with actual data where dates overlap
  const predictionDates = predictions.map(p => p.date);
  const actualDates = actualData.map(p => p.date);
  
  // Find matching dates
  for (let i = 0; i < predictionDates.length; i++) {
    const actualIndex = actualDates.indexOf(predictionDates[i]);
    if (actualIndex >= 0) {
      const error = Math.abs(predictions[i].price - actualData[actualIndex].close);
      totalError += error;
      availableComparisons++;
    }
  }
  
  const loss = availableComparisons > 0 ? totalError / availableComparisons : 999;
  
  // Calculate prediction stability (standard deviation of day-to-day changes)
  const dayChanges: number[] = [];
  for (let i = 1; i < predictions.length; i++) {
    const change = Math.abs(predictions[i].price - predictions[i-1].price) / predictions[i-1].price;
    dayChanges.push(change);
  }
  
  const avgChange = dayChanges.reduce((sum, val) => sum + val, 0) / dayChanges.length;
  const squaredDiffs = dayChanges.map(change => Math.pow(change - avgChange, 2));
  const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / squaredDiffs.length;
  const predictionStability = 1 - Math.sqrt(variance) * 100; // Higher is more stable
  
  // Calculate consistency score (how often the prediction direction matches actual direction)
  let correctDirections = 0;
  let totalDirections = 0;
  
  for (let i = 1; i < predictionDates.length; i++) {
    const prevDate = predictionDates[i-1];
    const currDate = predictionDates[i];
    
    const prevActualIndex = actualDates.indexOf(prevDate);
    const currActualIndex = actualDates.indexOf(currDate);
    
    if (prevActualIndex >= 0 && currActualIndex >= 0) {
      const predictedDirection = predictions[i].price > predictions[i-1].price;
      const actualDirection = actualData[currActualIndex].close > actualData[prevActualIndex].close;
      
      if (predictedDirection === actualDirection) {
        correctDirections++;
      }
      
      totalDirections++;
    }
  }
  
  const consistencyScore = totalDirections > 0 ? (correctDirections / totalDirections) * 100 : 0;
  
  // Calculate confidence based on prediction interval width
  const lastPrediction = predictions[predictions.length - 1];
  const intervalRatio = (lastPrediction.upper - lastPrediction.lower) / lastPrediction.price;
  const baseConfidence = 95 - (intervalRatio * 100);
  const confidence = Math.min(95, Math.max(60, baseConfidence));
  
  return {
    loss,
    confidence,
    consistencyScore,
    evaluationTime,
    predictionStability,
    algorithmName,
    parameterName: paramName,
    parameterValue: paramValue
  };
}

/**
 * Get default start date (3 years ago)
 */
function getDefaultStartDate(): string {
  const today = new Date();
  const threeYearsAgo = new Date(today);
  threeYearsAgo.setFullYear(today.getFullYear() - 3);
  return threeYearsAgo.toISOString().split('T')[0];
}

/**
 * Get default end date (yesterday)
 */
function getDefaultEndDate(): string {
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  return yesterday.toISOString().split('T')[0];
}

/**
 * Example usage:
 * 
 * // Run a tuning experiment for XGBoost numTrees parameter
 * const results = await runTuningExperiment(
 *   'xgboost',
 *   'numTrees',
 *   [10, 20, 50, 100],
 *   'AAPL',
 *   '2020-01-01',
 *   '2023-01-01'
 * );
 * 
 * // Results will contain evaluation metrics for each numTrees value
 * console.log(results);
 */ 