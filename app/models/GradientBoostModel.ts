import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import { PredictionModel } from './index';

export interface GradientBoostModelParams {
  numTrees?: number;
  maxDepth?: number;
  learningRate?: number;
  subsampleRatio?: number;
  minLeafSamples?: number;  // Minimum number of samples in a leaf node
  l2RegLambda?: number;     // L2 regularization parameter
  earlyStoppingRounds?: number; // Early stopping parameter
  validationSplit?: number; // Percentage of data to use for validation
  featureFraction?: number; // Fraction of features to consider per tree
  useHistoricalVolatility?: boolean; // Whether to use historical volatility in predictions
}

/**
 * Gradient Boosting model for stock price prediction
 * JavaScript implementation of gradient boosting algorithm
 */
export class GradientBoostModel implements PredictionModel {
  private numTrees: number;
  private maxDepth: number;
  private learningRate: number;
  private subsampleRatio: number;
  private minLeafSamples: number;
  private l2RegLambda: number;
  private earlyStoppingRounds: number;
  private validationSplit: number;
  private featureFraction: number;
  private useHistoricalVolatility: boolean;
  private trees: any[] = [];
  private initialPrediction: number = 0;
  private featureImportance: number[] = [];
  private validationErrors: number[] = [];

  constructor(params: GradientBoostModelParams = {}) {
    this.numTrees = params.numTrees || 150;
    this.maxDepth = params.maxDepth || 4;
    this.learningRate = params.learningRate || 0.03;
    this.subsampleRatio = params.subsampleRatio || 0.85;
    this.minLeafSamples = params.minLeafSamples || 5;
    this.l2RegLambda = params.l2RegLambda || 0.1;
    this.earlyStoppingRounds = params.earlyStoppingRounds || 10;
    this.validationSplit = params.validationSplit || 0.2;
    this.featureFraction = params.featureFraction || 0.8;
    this.useHistoricalVolatility = params.useHistoricalVolatility !== undefined ? params.useHistoricalVolatility : true;
    
    console.log('🔍 GradientBoostModel created with params:', 
      `numTrees=${this.numTrees}, maxDepth=${this.maxDepth}, learningRate=${this.learningRate}, ` +
      `minLeafSamples=${this.minLeafSamples}, l2RegLambda=${this.l2RegLambda}`);
  }

  /**
   * Train the Gradient Boosting model
   */
  async train(historicalData: StockDataPoint[]): Promise<any> {
    try {
      console.log(`Training Gradient Boosting model with ${historicalData.length} data points...`);
      
      // Input validation - exit early if data is insufficient
      if (!historicalData || historicalData.length < 30) {
        console.warn("Insufficient historical data for Gradient Boosting training");
        throw new Error("Insufficient training data");
      }

      // Ensure data is clean - filter out any records with non-finite values
      const cleanData = historicalData.filter(point => 
        isFinite(point.close) && 
        isFinite(point.open) && 
        isFinite(point.high) && 
        isFinite(point.low) && 
        isFinite(point.volume)
      );

      // If too much data was filtered, fail training
      if (cleanData.length < 30) {
        console.warn("Too many invalid data points for Gradient Boosting training");
        throw new Error("Too many invalid data points");
      }
      
      // Create features from historical data
      const features = this.extractFeatures(cleanData);
      
      // Create target variable (next day's price change percentage)
      const targets = [];
      for (let i = 0; i < cleanData.length - 1; i++) {
        const priceChange = (cleanData[i + 1].close - cleanData[i].close) / cleanData[i].close;
        if (isFinite(priceChange)) {
          targets.push(priceChange);
        } else {
          // Use zero change if calculation is invalid
          targets.push(0);
        }
      }
      
      if (features.length === 0 || targets.length === 0) {
        console.warn("Feature extraction failed for Gradient Boosting training");
        throw new Error("Feature extraction failed");
      }
      
      // Split data into training and validation sets
      const validationSize = Math.floor(features.length * this.validationSplit);
      const trainSize = features.length - validationSize;
      
      const trainFeatures = features.slice(0, trainSize);
      const trainTargets = targets.slice(0, trainSize);
      const validFeatures = features.slice(trainSize);
      const validTargets = targets.slice(trainSize);
      
      // Train using gradient boosting with early stopping
      this.initialPrediction = this.calculateMean(trainTargets);
      this.trees = [];
      this.validationErrors = [];
      this.featureImportance = new Array(trainFeatures[0].length).fill(0);
      
      // Initial predictions based on the mean
      let trainPredictions = new Array(trainSize).fill(this.initialPrediction);
      let validPredictions = new Array(validationSize).fill(this.initialPrediction);
      
      let bestValidationError = Infinity;
      let bestTreeCount = 0;
      let noImprovementCount = 0;
      
      // Build trees sequentially
      for (let i = 0; i < this.numTrees; i++) {
        // Calculate residuals (gradients)
        const residuals = trainTargets.map((actual, idx) => actual - trainPredictions[idx]);
        
        // Train a new tree
        const tree = this.trainTree(trainFeatures, residuals);
        this.trees.push(tree);
        
        // Update predictions and track feature importance
        this.updatePredictionsAndImportance(
          tree, 
          trainFeatures, 
          trainPredictions, 
          validFeatures, 
          validPredictions
        );
        
        // Calculate validation error
        const validationError = this.calculateMeanAbsoluteError(validPredictions, validTargets);
        this.validationErrors.push(validationError);
        
        // Check for early stopping
        if (validationError < bestValidationError) {
          bestValidationError = validationError;
          bestTreeCount = i + 1;
          noImprovementCount = 0;
        } else {
          noImprovementCount++;
        }
        
        // If no improvement for earlyStoppingRounds consecutive iterations, stop
        if (noImprovementCount >= this.earlyStoppingRounds && i >= 50) {
          console.log(`Early stopping at tree ${i+1}. Best tree count: ${bestTreeCount}`);
          // Keep only the best trees
          this.trees = this.trees.slice(0, bestTreeCount);
          break;
        }
      }
      
      // Normalize feature importance
      if (this.featureImportance.length > 0) {
        const totalImportance = this.featureImportance.reduce((sum, val) => sum + val, 0);
        if (totalImportance > 0) {
          this.featureImportance = this.featureImportance.map(val => val / totalImportance);
        }
      }
      
      console.log(`Gradient Boosting training complete with ${this.trees.length} trees`);
      return { 
        success: true, 
        numTrees: this.trees.length,
        featureImportance: this.featureImportance,
        validationError: bestValidationError
      };
    } catch (error) {
      console.error("Error in Gradient Boosting training:", error);
      throw error;
    }
  }

  /**
   * Generate predictions using the trained Gradient Boosting model
   */
  async predict(historicalData: StockDataPoint[], days: number = 30): Promise<PredictionPoint[]> {
    try {
      // Input validation
      if (!historicalData || historicalData.length < 30) {
        console.warn("Insufficient historical data for Gradient Boosting prediction");
        throw new Error("Insufficient prediction data");
      }
      
      // If model hasn't been trained yet, train it now
      if (!this.trees || this.trees.length === 0) {
        await this.train(historicalData);
      }

      // Ensure data is clean - filter out any records with non-finite values
      const cleanData = historicalData.filter(point => 
        isFinite(point.close) && 
        isFinite(point.open) && 
        isFinite(point.high) && 
        isFinite(point.low) && 
        isFinite(point.volume)
      );
      
      // Make predictions
      const predictions: PredictionPoint[] = [];
      const lastDate = new Date(cleanData[cleanData.length - 1].date);
      
      // Start with last known price
      let currentPrice = cleanData[cleanData.length - 1].close;
      
      // Calculate recent trend for momentum adjustment
      const recentPrices = cleanData.slice(-30).map(d => d.close);
      const recentTrend = this.calculateRecentTrend(recentPrices);
      
      // Calculate volatility for confidence intervals
      const volatility = this.calculateVolatility(cleanData);
      const safeVolatility = isFinite(volatility) && volatility > 0 ? volatility : 0.01;
      
      // Get market regime features (bull/bear/sideways)
      const marketRegime = this.detectMarketRegime(cleanData);
      
      // Get the latest features
      const features = this.extractFeatures(cleanData);
      let currentFeatures = features[features.length - 1];
      
      // Determine if we're in a high volatility environment
      const isHighVolatility = safeVolatility > 0.015; // 1.5% daily volatility threshold
      
      // Calculate price levels (support/resistance)
      const priceLevels = this.calculatePriceLevels(cleanData);
      
      // Calculate predictive metrics
      const metrics = {
        rsi: this.calculateRSI(cleanData),
        momentum: this.calculateMomentum(cleanData),
        trendStrength: Math.abs(recentTrend) * 10, // Scale up for easier use
        averageVolume: this.calculateAverageVolume(cleanData),
        priceToSMA50: currentPrice / this.calculateSMA(cleanData, 50),
        priceToSMA200: currentPrice / this.calculateSMA(cleanData, 200)
      };
      
      // Create synthetic data for multiple simulation paths
      const numSimulations = 20;
      const simulationPaths: number[][] = [];
      
      // Initialize simulations with current price
      for (let i = 0; i < numSimulations; i++) {
        simulationPaths.push([currentPrice]);
      }
      
      // Run the simulations
      for (let day = 1; day <= days; day++) {
        const predictionDate = new Date(lastDate);
        predictionDate.setDate(lastDate.getDate() + day);
        const formattedDate = predictionDate.toISOString().split('T')[0];
        
        // For each simulation path
        const dayPrices: number[] = [];
        
        for (let sim = 0; sim < numSimulations; sim++) {
          // Get the previous day's price for this simulation
          const prevPrice = simulationPaths[sim][day - 1];
          
          // Predict price change using gradient boosting model
          let predictedChange = this.predictWithGradientBoosting(currentFeatures);
          
          // Apply adjustments based on market regime
          if (marketRegime === 'bull' && predictedChange < 0) {
            predictedChange *= 0.7; // Dampen negative moves in bull market
          } else if (marketRegime === 'bear' && predictedChange > 0) {
            predictedChange *= 0.7; // Dampen positive moves in bear market
          }
          
          // Add momentum effect
          predictedChange += recentTrend * 0.1; 
          
          // Add mean reversion effect for longer predictions
          if (day > 10) {
            const meanReversionStrength = Math.min(0.001 * (day - 10), 0.01);
            if (metrics.priceToSMA200 > 1.1) {
              // Price is well above 200-day SMA, add downward pressure
              predictedChange -= meanReversionStrength;
            } else if (metrics.priceToSMA200 < 0.9) {
              // Price is well below 200-day SMA, add upward pressure
              predictedChange += meanReversionStrength;
            }
          }
          
          // Add random noise based on historical volatility
          // More noise for high volatility periods and longer predictions
          const noiseLevel = safeVolatility * Math.sqrt(Math.min(day, 10)) / 5;
          const noise = this.generateNormalRandom() * noiseLevel;
          
          // In high volatility environments, add more noise
          const volatilityMultiplier = isHighVolatility ? 1.5 : 1.0;
          
          // Calculate the new price
          let newPrice = prevPrice * (1 + predictedChange + noise * volatilityMultiplier);
          
          // Consider price levels (support/resistance)
          newPrice = this.adjustForPriceLevels(newPrice, priceLevels);
          
          // Safety cap to prevent extreme movements
          const maxDailyMove = 0.05 * Math.sqrt(day); // 5% base, scaled by sqrt of days
          const cappedChange = Math.max(-maxDailyMove, Math.min(maxDailyMove, (newPrice / prevPrice) - 1));
          newPrice = prevPrice * (1 + cappedChange);
          
          // Ensure price is positive
          newPrice = Math.max(0.01, newPrice);
          
          // Store the predicted price for this simulation
          simulationPaths[sim].push(newPrice);
          dayPrices.push(newPrice);
          
          // Only update features for the main simulation path (first one)
          if (sim === 0 && day < days) {
            // Create a synthetic data point for feature update
            const syntheticPoint: StockDataPoint = {
              date: formattedDate,
              timestamp: lastDate.getTime() + day * 86400000, // Add days in milliseconds
              open: newPrice * 0.998, // Approximate open price
              high: newPrice * 1.005, // Approximate high
              low: newPrice * 0.995, // Approximate low
              close: newPrice,
              volume: this.estimateVolume(cleanData, metrics.averageVolume, cappedChange)
            };
            
            // Update features for next prediction
            try {
              const updatedFeatures = this.updateFeaturesForNextPrediction(
                currentFeatures, 
                newPrice, 
                [...cleanData, syntheticPoint]
              );
              currentFeatures = updatedFeatures.map(f => isFinite(f) ? f : 0);
            } catch (err) {
              // If feature update fails, keep previous features with slight adjustment
              console.warn("Failed to update features for next prediction step", err);
              currentFeatures = currentFeatures.map(f => f * (1 + 0.01 * this.generateNormalRandom()));
            }
          }
        }
        
        // Calculate statistics across simulation paths for this day
        dayPrices.sort((a, b) => a - b);
        const medianPrice = dayPrices[Math.floor(dayPrices.length / 2)];
        const lowerBound = dayPrices[Math.floor(dayPrices.length * 0.1)]; // 10th percentile
        const upperBound = dayPrices[Math.floor(dayPrices.length * 0.9)]; // 90th percentile
        
        // Add prediction with bounds
        predictions.push({
          date: formattedDate,
          price: medianPrice,
          upper: upperBound,
          lower: lowerBound,
          algorithmUsed: 'gradientboost'
        });
      }
      
      return predictions;
    } catch (error) {
      console.error("Error in Gradient Boosting prediction:", error);
      throw error;
    }
  }

  /**
   * Save the model - in this case, just storing tree structure
   */
  async saveModel(modelName: string): Promise<any> {
    try {
      // In a real app, save the trees to a file or database
      return { success: true, message: `Gradient Boosting model '${modelName}' saved successfully` };
    } catch (error) {
      console.error(`Error saving Gradient Boosting model: ${error}`);
      throw error;
    }
  }

  /**
   * Load a previously saved model
   */
  async loadModel(modelName: string): Promise<void> {
    try {
      // In a real app, load trees from a file or database
      console.log(`Loading Gradient Boosting model '${modelName}'`);
      // Currently, this is a placeholder for future implementation
    } catch (error) {
      console.error(`Error loading Gradient Boosting model: ${error}`);
      throw error;
    }
  }
  
  /**
   * Extract features for Gradient Boosting model
   */
  private extractFeatures(data: StockDataPoint[]): number[][] {
    if (!data || data.length < 21) {
      return [];
    }
    
    const features = [];
    
    // We need at least 20 previous days for feature calculation
    for (let i = 20; i < data.length; i++) {
      const currentPoint = data[i];
      const prevPoint = data[i - 1];
      const prev5Point = data[i - 5];
      const prev10Point = data[i - 10];
      const prev20Point = data[i - 20];
      
      // Price-based features
      const priceChange1d = (currentPoint.close - prevPoint.close) / prevPoint.close;
      const priceChange5d = (currentPoint.close - prev5Point.close) / prev5Point.close;
      const priceChange10d = (currentPoint.close - prev10Point.close) / prev10Point.close;
      const priceChange20d = (currentPoint.close - prev20Point.close) / prev20Point.close;
      
      // Moving averages
      const ma5 = this.calculateSMA(data.slice(i - 5, i + 1), 5);
      const ma10 = this.calculateSMA(data.slice(i - 10, i + 1), 10);
      const ma20 = this.calculateSMA(data.slice(i - 20, i + 1), 20);
      const ma50 = i >= 50 ? this.calculateSMA(data.slice(i - 50, i + 1), 50) : ma20;
      
      // Moving average crossovers
      const ma5CrossMa20 = ma5 / ma20 - 1;
      const ma10CrossMa50 = i >= 50 ? ma10 / ma50 - 1 : 0;
      
      // Volatility features
      const volatility5d = this.calculateVolatility(data.slice(i - 5, i + 1));
      const volatility10d = this.calculateVolatility(data.slice(i - 10, i + 1));
      const volatility20d = this.calculateVolatility(data.slice(i - 20, i + 1));
      
      // Volume features
      const volumeChange1d = (currentPoint.volume - prevPoint.volume) / (prevPoint.volume || 1);
      const volumeChange5d = (currentPoint.volume - prev5Point.volume) / (prev5Point.volume || 1);
      const volumeAvg10d = this.calculateAverageVolume(data.slice(i - 10, i + 1));
      const relativeVolume = currentPoint.volume / (volumeAvg10d || 1);
      
      // Price range features
      const dayRange = (currentPoint.high - currentPoint.low) / currentPoint.low;
      const weekRange = (
        Math.max(...data.slice(i - 5, i + 1).map(d => d.high)) - 
        Math.min(...data.slice(i - 5, i + 1).map(d => d.low))
      ) / Math.min(...data.slice(i - 5, i + 1).map(d => d.low));
      
      // Technical indicators
      const rsi = this.calculateRSI(data.slice(0, i + 1));
      const rsiDiff = i >= 5 ? rsi - this.calculateRSI(data.slice(0, i - 4)) : 0;
      const momentum5d = this.calculateMomentum(data.slice(0, i + 1), 5);
      const momentum10d = this.calculateMomentum(data.slice(0, i + 1), 10);
      
      // Gap features
      const gapUp = (currentPoint.open - prevPoint.close) / prevPoint.close;
      const highLowRatio = currentPoint.high / currentPoint.low;
      
      // Price relative to historical ranges
      const month_high = Math.max(...data.slice(Math.max(0, i - 20), i + 1).map(d => d.high));
      const month_low = Math.min(...data.slice(Math.max(0, i - 20), i + 1).map(d => d.low));
      const price_to_month_high = currentPoint.close / month_high;
      const price_to_month_low = currentPoint.close / month_low;
      
      // Derived features
      const priceAcceleration = i >= 10 ? priceChange5d - this.calculateMomentum(data.slice(0, i - 4), 5) : 0;
      const trendConsistency = this.calculateTrendConsistency(data.slice(i - 10, i + 1));
      
      // Create feature vector - make sure all values are finite
      const featureVector = [
        isFinite(priceChange1d) ? priceChange1d : 0,
        isFinite(priceChange5d) ? priceChange5d : 0,
        isFinite(priceChange10d) ? priceChange10d : 0,
        isFinite(priceChange20d) ? priceChange20d : 0,
        isFinite(ma5) ? ma5 / currentPoint.close - 1 : 0,
        isFinite(ma10) ? ma10 / currentPoint.close - 1 : 0,
        isFinite(ma20) ? ma20 / currentPoint.close - 1 : 0,
        isFinite(ma50) ? ma50 / currentPoint.close - 1 : 0,
        isFinite(ma5CrossMa20) ? ma5CrossMa20 : 0,
        isFinite(ma10CrossMa50) ? ma10CrossMa50 : 0,
        isFinite(volatility5d) ? volatility5d : 0,
        isFinite(volatility10d) ? volatility10d : 0,
        isFinite(volatility20d) ? volatility20d : 0,
        isFinite(volumeChange1d) ? volumeChange1d : 0,
        isFinite(volumeChange5d) ? volumeChange5d : 0,
        isFinite(relativeVolume) ? relativeVolume : 1,
        isFinite(dayRange) ? dayRange : 0,
        isFinite(weekRange) ? weekRange : 0,
        isFinite(rsi) ? rsi / 100 : 0.5, // Normalize RSI to 0-1
        isFinite(rsiDiff) ? rsiDiff / 100 : 0,
        isFinite(momentum5d) ? momentum5d : 0,
        isFinite(momentum10d) ? momentum10d : 0,
        isFinite(gapUp) ? gapUp : 0,
        isFinite(highLowRatio) ? highLowRatio - 1 : 0,
        isFinite(price_to_month_high) ? price_to_month_high : 0.9,
        isFinite(price_to_month_low) ? price_to_month_low : 1.1,
        isFinite(priceAcceleration) ? priceAcceleration : 0,
        isFinite(trendConsistency) ? trendConsistency : 0
      ];
      
      features.push(featureVector);
    }
    
    return features;
  }
  
  /**
   * Calculate trend consistency (how many days move in the same direction)
   */
  private calculateTrendConsistency(data: StockDataPoint[]): number {
    if (!data || data.length < 5) {
      return 0;
    }
    
    let upDays = 0;
    let downDays = 0;
    
    for (let i = 1; i < data.length; i++) {
      if (data[i].close > data[i-1].close) {
        upDays++;
      } else if (data[i].close < data[i-1].close) {
        downDays++;
      }
    }
    
    const totalDays = data.length - 1;
    const dominantTrendRatio = Math.max(upDays, downDays) / totalDays;
    
    return dominantTrendRatio;
  }
  
  /**
   * Update feature vector for the next prediction step
   */
  private updateFeaturesForNextPrediction(
    currentFeatures: number[], 
    predictedPrice: number, 
    historicalData: StockDataPoint[]
  ): number[] {
    // Create a synthetic data point using the predicted price
    const lastPoint = historicalData[historicalData.length - 1];
    const syntheticPoint: StockDataPoint = {
      date: 'prediction',
      timestamp: lastPoint.timestamp + 86400000, // Next day in milliseconds
      open: predictedPrice * 0.99, // Approximate open price
      high: predictedPrice * 1.01, // Approximate high
      low: predictedPrice * 0.99, // Approximate low
      close: predictedPrice,
      volume: lastPoint.volume // Reuse last volume as approximation
    };
    
    // Add the synthetic point to the data for feature calculation
    const extendedData = [...historicalData, syntheticPoint];
    
    // Extract new features using the extended data
    const newFeatures = this.extractFeatures(extendedData);
    
    // Return the latest feature vector
    return newFeatures[newFeatures.length - 1];
  }
  
  /**
   * Train gradient boosting model
   */
  private trainGradientBoosting(features: number[][], targets: number[]): any[] {
    const trees = [];
    const numSamples = features.length;
    
    // Initialize predictions with the mean target value
    const initialPrediction = this.calculateMean(targets);
    this.initialPrediction = initialPrediction;
    
    // Initial predictions are just the mean value
    let currentPredictions = new Array(numSamples).fill(initialPrediction);
    
    // Build trees sequentially
    for (let i = 0; i < this.numTrees; i++) {
      // Calculate residuals (difference between actual and predicted values)
      const residuals = targets.map((actual, idx) => actual - currentPredictions[idx]);
      
      // Subsample the data (stochastic gradient boosting)
      const { subsampledFeatures, subsampledResiduals } = this.subsampleData(features, residuals);
      
      // Build a regression tree on the residuals
      const tree = this.buildRegressionTree(subsampledFeatures, subsampledResiduals, 0, this.maxDepth);
      
      // Update predictions
      for (let j = 0; j < numSamples; j++) {
        const prediction = this.predictWithTree(tree, features[j]);
        currentPredictions[j] += prediction * this.learningRate;
      }
      
      trees.push(tree);
    }
    
    return trees;
  }
  
  /**
   * Subsample data for stochastic gradient boosting
   */
  private subsampleData(
    features: number[][], 
    targets: number[]
  ): { subsampledFeatures: number[][], subsampledResiduals: number[] } {
    const numSamples = features.length;
    const sampleSize = Math.floor(numSamples * this.subsampleRatio);
    
    // Create an array of indices and shuffle it
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    // Take the first sampleSize indices
    const selectedIndices = indices.slice(0, sampleSize);
    
    // Create subsampled data
    const subsampledFeatures = selectedIndices.map(idx => features[idx]);
    const subsampledResiduals = selectedIndices.map(idx => targets[idx]);
    
    return { subsampledFeatures, subsampledResiduals };
  }
  
  /**
   * Build a regression tree recursively
   */
  private buildRegressionTree(
    features: number[][], 
    targets: number[], 
    depth: number, 
    maxDepth: number
  ): any {
    // Terminal conditions
    if (depth >= maxDepth || features.length <= this.minLeafSamples) {
      // Apply L2 regularization to leaf values
      const leafValue = this.calculateMean(targets);
      const regularizedValue = leafValue / (1 + this.l2RegLambda);
      
      return { 
        isLeaf: true, 
        value: regularizedValue,
        samples: targets.length
      };
    }
    
    // Find the best split
    const bestSplit = this.findBestSplit(features, targets);
    
    if (bestSplit.gain <= 0) {
      return { 
        isLeaf: true, 
        value: this.calculateMean(targets),
        samples: targets.length
      };
    }
    
    // Split the data
    const { leftFeatures, leftTargets, rightFeatures, rightTargets } = 
      this.splitData(features, targets, bestSplit.featureIndex, bestSplit.threshold);
    
    // If split doesn't actually divide the data, or one side has too few samples, return a leaf
    if (leftFeatures.length < this.minLeafSamples || rightFeatures.length < this.minLeafSamples) {
      return { 
        isLeaf: true, 
        value: this.calculateMean(targets),
        samples: targets.length
      };
    }
    
    // Recursively build subtrees
    const leftSubtree = this.buildRegressionTree(
      leftFeatures, 
      leftTargets, 
      depth + 1, 
      maxDepth
    );
    
    const rightSubtree = this.buildRegressionTree(
      rightFeatures, 
      rightTargets, 
      depth + 1, 
      maxDepth
    );
    
    // Return the decision node
    return {
      isLeaf: false,
      featureIndex: bestSplit.featureIndex,
      threshold: bestSplit.threshold,
      gain: bestSplit.gain,
      samples: targets.length,
      left: leftSubtree,
      right: rightSubtree
    };
  }
  
  /**
   * Find the best split for a regression tree node
   */
  private findBestSplit(features: number[][], targets: number[]): { featureIndex: number, threshold: number, gain: number } {
    let bestGain = -Infinity;
    let bestFeatureIndex = -1;
    let bestThreshold = 0;
    
    // Calculate variance of current node (for gain calculation)
    const nodeVariance = this.calculateVariance(targets);
    
    // Calculate feature subset to consider (stochastic)
    const numFeatures = features[0].length;
    const featuresToConsider = Math.max(1, Math.floor(numFeatures * this.featureFraction));
    
    // Create a randomly shuffled array of feature indices
    const featureIndices = Array.from({ length: numFeatures }, (_, i) => i);
    this.shuffleArray(featureIndices);
    
    // Only consider a subset of features (feature subsampling)
    const selectedFeatures = featureIndices.slice(0, featuresToConsider);
    
    // Try each selected feature
    for (const featureIndex of selectedFeatures) {
      // Get values for this feature
      const values = features.map(f => f[featureIndex]);
      
      // Determine number of thresholds to try based on dataset size
      const numThresholds = Math.min(
        20, 
        Math.max(5, Math.floor(Math.sqrt(values.length)))
      );
      
      // Generate potential thresholds using quantiles
      const sortedValues = [...values].sort((a, b) => a - b);
      const thresholds = [];
      
      for (let i = 1; i < numThresholds; i++) {
        const idx = Math.floor((i / numThresholds) * sortedValues.length);
        thresholds.push(sortedValues[idx]);
      }
      
      // Add some additional random thresholds
      const randomThresholds = Array.from(
        { length: Math.min(5, Math.floor(numThresholds / 2)) }, 
        () => sortedValues[Math.floor(Math.random() * sortedValues.length)]
      );
      
      thresholds.push(...randomThresholds);
      
      // Try each threshold
      for (const threshold of thresholds) {
        // Split the data
        const { leftTargets, rightTargets } = this.splitTargets(features, targets, featureIndex, threshold);
        
        // Skip if split is too imbalanced or below minimum samples
        if (leftTargets.length < this.minLeafSamples || rightTargets.length < this.minLeafSamples) {
          continue;
        }
        
        // Calculate gain (reduction in variance with regularization)
        const leftVariance = this.calculateVariance(leftTargets);
        const rightVariance = this.calculateVariance(rightTargets);
        
        const leftWeight = leftTargets.length / targets.length;
        const rightWeight = rightTargets.length / targets.length;
        
        // Apply regularization term to penalize complex trees
        const complexity = this.l2RegLambda * (1.0 / leftTargets.length + 1.0 / rightTargets.length);
        
        const weightedVariance = leftWeight * leftVariance + rightWeight * rightVariance;
        const gain = nodeVariance - weightedVariance - complexity;
        
        if (gain > bestGain) {
          bestGain = gain;
          bestFeatureIndex = featureIndex;
          bestThreshold = threshold;
        }
      }
    }
    
    return { 
      featureIndex: bestFeatureIndex, 
      threshold: bestThreshold, 
      gain: bestGain 
    };
  }
  
  /**
   * Shuffle array in-place using Fisher-Yates algorithm
   */
  private shuffleArray(array: any[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
  
  /**
   * Split targets based on a feature and threshold
   */
  private splitTargets(
    features: number[][], 
    targets: number[], 
    featureIndex: number, 
    threshold: number
  ): { leftTargets: number[], rightTargets: number[] } {
    const leftTargets: number[] = [];
    const rightTargets: number[] = [];
    
    for (let i = 0; i < features.length; i++) {
      if (features[i][featureIndex] <= threshold) {
        leftTargets.push(targets[i]);
      } else {
        rightTargets.push(targets[i]);
      }
    }
    
    return { leftTargets, rightTargets };
  }
  
  /**
   * Split data based on a feature and threshold
   */
  private splitData(
    features: number[][], 
    targets: number[], 
    featureIndex: number, 
    threshold: number
  ): { 
    leftFeatures: number[][], 
    leftTargets: number[], 
    rightFeatures: number[][], 
    rightTargets: number[] 
  } {
    const leftFeatures: number[][] = [];
    const leftTargets: number[] = [];
    const rightFeatures: number[][] = [];
    const rightTargets: number[] = [];
    
    for (let i = 0; i < features.length; i++) {
      if (features[i][featureIndex] <= threshold) {
        leftFeatures.push(features[i]);
        leftTargets.push(targets[i]);
      } else {
        rightFeatures.push(features[i]);
        rightTargets.push(targets[i]);
      }
    }
    
    return { leftFeatures, leftTargets, rightFeatures, rightTargets };
  }
  
  /**
   * Calculate mean of an array
   */
  private calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    const sum = values.reduce((acc, val) => acc + val, 0);
    return sum / values.length;
  }
  
  /**
   * Calculate variance of an array
   */
  private calculateVariance(values: number[]): number {
    if (values.length <= 1) return 0;
    
    const mean = this.calculateMean(values);
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return this.calculateMean(squaredDiffs);
  }
  
  /**
   * Predict using a single decision tree
   */
  private predictWithTree(tree: any, features: number[]): number {
    if (tree.isLeaf) {
      return tree.value;
    }
    
    // Protect against features not having the required index
    if (tree.featureIndex >= features.length) {
      return tree.left.isLeaf ? tree.left.value : tree.right.value;
    }
    
    if (features[tree.featureIndex] <= tree.threshold) {
      return this.predictWithTree(tree.left, features);
    } else {
      return this.predictWithTree(tree.right, features);
    }
  }
  
  /**
   * Predict using the gradient boosting model
   */
  private predictWithGradientBoosting(features: number[]): number {
    if (!this.trees || this.trees.length === 0) {
      return 0;
    }
    
    // Start with initial prediction (mean of training data)
    let prediction = this.initialPrediction;
    
    // Add contributions from each tree
    for (const tree of this.trees) {
      prediction += this.predictWithTree(tree, features) * this.learningRate;
    }
    
    return prediction;
  }
  
  /**
   * Calculate Simple Moving Average
   */
  private calculateSMA(data: StockDataPoint[], window: number): number {
    if (!data || data.length === 0 || window <= 0) {
      return 0;
    }
    
    const effectiveWindow = Math.min(window, data.length);
    const windowData = data.slice(Math.max(0, data.length - effectiveWindow));
    
    if (windowData.length === 0) {
      return data[data.length - 1]?.close || 0;
    }
    
    const sum = windowData.reduce((sum, point) => sum + (isFinite(point.close) ? point.close : 0), 0);
    const result = sum / windowData.length;
    
    return isFinite(result) ? result : 0;
  }
  
  /**
   * Calculate price momentum
   */
  private calculateMomentum(data: StockDataPoint[], period: number = 5): number {
    if (!data || data.length <= period) {
      return 0;
    }
    
    const currentPrice = data[data.length - 1].close;
    const previousPrice = data[data.length - 1 - period].close;
    
    const momentum = (currentPrice - previousPrice) / previousPrice;
    return isFinite(momentum) ? momentum : 0;
  }
  
  /**
   * Calculate Relative Strength Index (RSI)
   */
  private calculateRSI(data: StockDataPoint[], period: number = 14): number {
    if (!data || data.length <= period + 1) {
      return 50; // Default to neutral RSI
    }
    
    let gains = 0;
    let losses = 0;
    
    // Calculate average gains and losses
    for (let i = data.length - period; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close;
      if (change >= 0) {
        gains += change;
      } else {
        losses -= change; // Make positive
      }
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    
    if (avgLoss === 0) {
      return 100; // All gains, no losses
    }
    
    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));
    
    return isFinite(rsi) ? rsi : 50;
  }
  
  /**
   * Calculate price volatility
   */
  private calculateVolatility(data: StockDataPoint[]): number {
    if (!data || data.length < 2) {
      return 0.01; // Default low volatility
    }
    
    // Calculate daily returns
    const returns = [];
    for (let i = 1; i < data.length; i++) {
      const dailyReturn = (data[i].close - data[i - 1].close) / data[i - 1].close;
      if (isFinite(dailyReturn)) {
        returns.push(dailyReturn);
      }
    }
    
    if (returns.length === 0) {
      return 0.01;
    }
    
    // Calculate standard deviation of returns
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const squaredDiffs = returns.map(ret => Math.pow(ret - mean, 2));
    const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    return isFinite(stdDev) ? stdDev : 0.01;
  }
  
  /**
   * Calculate recent price trend direction
   */
  private calculateRecentTrend(prices: number[]): number {
    if (!prices || prices.length < 5) {
      return 0; // No trend
    }
    
    // Simple linear regression
    const n = prices.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    
    const sumX = indices.reduce((sum, x) => sum + x, 0);
    const sumY = prices.reduce((sum, y) => sum + y, 0);
    const sumXY = indices.reduce((sum, x, i) => sum + x * prices[i], 0);
    const sumXX = indices.reduce((sum, x) => sum + x * x, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    
    // Normalize the slope relative to the average price
    const avgPrice = sumY / n;
    const normalizedSlope = slope / avgPrice;
    
    return isFinite(normalizedSlope) ? normalizedSlope : 0;
  }

  /**
   * Update predictions and track feature importance
   */
  private updatePredictionsAndImportance(
    tree: any,
    trainFeatures: number[][],
    trainPredictions: number[],
    validFeatures: number[][],
    validPredictions: number[]
  ): void {
    // Update training predictions
    for (let i = 0; i < trainFeatures.length; i++) {
      const prediction = this.predictWithTree(tree, trainFeatures[i]);
      trainPredictions[i] += prediction * this.learningRate;
    }
    
    // Update validation predictions
    for (let i = 0; i < validFeatures.length; i++) {
      const prediction = this.predictWithTree(tree, validFeatures[i]);
      validPredictions[i] += prediction * this.learningRate;
    }
    
    // Update feature importance based on tree usage
    this.updateFeatureImportance(tree);
  }
  
  /**
   * Calculate Mean Absolute Error
   */
  private calculateMeanAbsoluteError(predictions: number[], targets: number[]): number {
    if (predictions.length !== targets.length || predictions.length === 0) {
      return Infinity;
    }
    
    let sumErrors = 0;
    for (let i = 0; i < predictions.length; i++) {
      sumErrors += Math.abs(predictions[i] - targets[i]);
    }
    
    return sumErrors / predictions.length;
  }

  /**
   * Train a single regression tree with regularization
   */
  private trainTree(features: number[][], targets: number[]): any {
    // Subsample the data (stochastic gradient boosting)
    const { subsampledFeatures, subsampledResiduals } = this.subsampleData(features, targets);
    
    // Build a regression tree on the targets
    return this.buildRegressionTree(
      subsampledFeatures, 
      subsampledResiduals, 
      0, 
      this.maxDepth
    );
  }
  
  /**
   * Update feature importance based on tree structure
   */
  private updateFeatureImportance(tree: any, depth: number = 0, weight: number = 1.0): void {
    if (!tree || tree.isLeaf) {
      return;
    }
    
    // Increase importance for the feature used in this split
    // Importance is weighted by depth (earlier splits are more important)
    const depthWeight = Math.pow(0.8, depth); // Decay factor for depth
    const featureIdx = tree.featureIndex;
    
    if (featureIdx >= 0 && featureIdx < this.featureImportance.length) {
      this.featureImportance[featureIdx] += weight * depthWeight;
    }
    
    // Recursively update importance for subtrees
    const leftWeight = tree.left && !tree.left.isLeaf ? 0.6 : 0.3;
    const rightWeight = tree.right && !tree.right.isLeaf ? 0.6 : 0.3;
    
    this.updateFeatureImportance(tree.left, depth + 1, weight * leftWeight);
    this.updateFeatureImportance(tree.right, depth + 1, weight * rightWeight);
  }

  /**
   * Detect market regime (bull, bear, or sideways)
   */
  private detectMarketRegime(data: StockDataPoint[]): 'bull' | 'bear' | 'sideways' {
    if (data.length < 50) {
      return 'sideways';
    }
    
    // Look at price relative to moving averages
    const currentPrice = data[data.length - 1].close;
    const sma50 = this.calculateSMA(data, 50);
    const sma200 = this.calculateSMA(data, 200);
    
    // Check recent trend
    const shortTrend = this.calculateRecentTrend(data.slice(-20).map(d => d.close));
    const mediumTrend = this.calculateRecentTrend(data.slice(-50).map(d => d.close));
    
    // Determine regime
    if (currentPrice > sma50 && sma50 > sma200 && shortTrend > 0 && mediumTrend > 0) {
      return 'bull';
    } else if (currentPrice < sma50 && sma50 < sma200 && shortTrend < 0 && mediumTrend < 0) {
      return 'bear';
    } else {
      return 'sideways';
    }
  }
  
  /**
   * Calculate price levels (support and resistance)
   */
  private calculatePriceLevels(data: StockDataPoint[]): { support: number[], resistance: number[] } {
    if (data.length < 30) {
      return { support: [], resistance: [] };
    }
    
    const currentPrice = data[data.length - 1].close;
    const prices = data.slice(-60).map(d => d.close);
    
    // Find local minima (support) and maxima (resistance)
    const support: number[] = [];
    const resistance: number[] = [];
    
    // Simple approach: look for local extrema
    for (let i = 5; i < prices.length - 5; i++) {
      const window = prices.slice(i - 5, i + 6);
      const center = prices[i];
      
      // Check if center is a local minimum
      if (window.every(p => p >= center)) {
        support.push(center);
      }
      
      // Check if center is a local maximum
      if (window.every(p => p <= center)) {
        resistance.push(center);
      }
    }
    
    // Filter to only keep levels that are reasonably close to current price
    return {
      support: support.filter(s => s < currentPrice && s > currentPrice * 0.8),
      resistance: resistance.filter(r => r > currentPrice && r < currentPrice * 1.2)
    };
  }
  
  /**
   * Adjust price based on support and resistance levels
   */
  private adjustForPriceLevels(
    price: number, 
    levels: { support: number[], resistance: number[] }
  ): number {
    // If no levels, return original price
    if (levels.support.length === 0 && levels.resistance.length === 0) {
      return price;
    }
    
    // Check for nearby support
    for (const support of levels.support) {
      // If price is slightly below support, pull it up a bit
      if (price < support && price > support * 0.97) {
        // The closer to support, the stronger the pull
        const pullFactor = 1 - (support - price) / (support * 0.03);
        price = price * (1 - pullFactor) + support * pullFactor;
        break;
      }
    }
    
    // Check for nearby resistance
    for (const resistance of levels.resistance) {
      // If price is slightly above resistance, pull it down a bit
      if (price > resistance && price < resistance * 1.03) {
        // The closer to resistance, the stronger the pull
        const pullFactor = 1 - (price - resistance) / (resistance * 0.03);
        price = price * (1 - pullFactor) + resistance * pullFactor;
        break;
      }
    }
    
    return price;
  }
  
  /**
   * Estimate volume for next prediction based on price change
   */
  private estimateVolume(
    data: StockDataPoint[], 
    averageVolume: number, 
    priceChange: number
  ): number {
    // Higher absolute price changes typically come with higher volumes
    const changeMultiplier = 1 + Math.abs(priceChange) * 5;
    
    // Add some randomness
    const randomFactor = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
    
    return averageVolume * changeMultiplier * randomFactor;
  }
  
  /**
   * Calculate average volume over the last n days
   */
  private calculateAverageVolume(data: StockDataPoint[], days: number = 20): number {
    if (data.length < days) {
      return data.reduce((sum, d) => sum + d.volume, 0) / data.length;
    }
    
    const recentVolumes = data.slice(-days).map(d => d.volume);
    return recentVolumes.reduce((sum, v) => sum + v, 0) / recentVolumes.length;
  }
  
  /**
   * Generate a normally distributed random number
   */
  private generateNormalRandom(): number {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0;
  }
} 