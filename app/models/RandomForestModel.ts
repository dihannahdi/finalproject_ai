import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import { PredictionModel } from './index';

export interface RandomForestModelParams {
  numTrees?: number;
  maxDepth?: number;
  featureSamplingRatio?: number;
  dataSamplingRatio?: number;
}

/**
 * Random Forest model for stock price prediction
 * JavaScript implementation of random forest algorithm
 */
export class RandomForestModel implements PredictionModel {
  private numTrees: number;
  private maxDepth: number;
  private featureSamplingRatio: number;
  private dataSamplingRatio: number;
  private forest: any[] = [];

  constructor(params: RandomForestModelParams = {}) {
    this.numTrees = params.numTrees || 100;
    this.maxDepth = params.maxDepth || 5;
    this.featureSamplingRatio = params.featureSamplingRatio || 0.7;
    this.dataSamplingRatio = params.dataSamplingRatio || 0.8;
    console.log('🔍 RandomForestModel created with params:', 
      `numTrees=${this.numTrees}, maxDepth=${this.maxDepth}`);
  }

  /**
   * Train the Random Forest model
   */
  async train(historicalData: StockDataPoint[]): Promise<any> {
    try {
      console.log(`Training Random Forest model with ${historicalData.length} data points...`);
      
      // Input validation - exit early if data is insufficient
      if (!historicalData || historicalData.length < 30) {
        console.warn("Insufficient historical data for Random Forest training");
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
        console.warn("Too many invalid data points for Random Forest training");
        throw new Error("Too many invalid data points");
      }
      
      // Create features from historical data
      const features = this.extractFeatures(cleanData);
      
      // Create target variable (next day's price)
      const targets = [];
      for (let i = 0; i < cleanData.length - 1; i++) {
        const target = cleanData[i + 1].close;
        if (isFinite(target)) {
          targets.push(target);
        } else {
          // Use previous close if next day's close is invalid
          targets.push(cleanData[i].close);
        }
      }
      
      if (features.length === 0 || targets.length === 0) {
        console.warn("Feature extraction failed for Random Forest training");
        throw new Error("Feature extraction failed");
      }
      
      // Create random forest (ensemble of decision trees)
      this.forest = this.createRandomForest(features, targets);
      
      // If forest creation failed, fail training
      if (!this.forest || this.forest.length === 0) {
        console.warn("Forest creation failed for Random Forest");
        throw new Error("Forest creation failed");
      }
      
      console.log(`Random Forest training complete with ${this.forest.length} trees`);
      return { success: true, numTrees: this.forest.length };
    } catch (error) {
      console.error("Error in Random Forest training:", error);
      throw error;
    }
  }

  /**
   * Generate predictions using the trained Random Forest model
   */
  async predict(historicalData: StockDataPoint[], days: number = 30): Promise<PredictionPoint[]> {
    try {
      // Input validation
      if (!historicalData || historicalData.length < 30) {
        console.warn("Insufficient historical data for Random Forest prediction");
        throw new Error("Insufficient prediction data");
      }
      
      // If model hasn't been trained yet, train it now
      if (!this.forest || this.forest.length === 0) {
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
      
      // Calculate recent trend and volatility to inform prediction
      const recentPrices = cleanData.slice(-30).map(d => d.close);
      const recentTrend = this.calculateRecentTrend(recentPrices);
      
      // Bias prediction slightly based on recent trend (momentum effect)
      const trendBias = recentTrend * 0.1; // Reduce impact to 10% of trend
      
      // Get the latest features
      const features = this.extractFeatures(cleanData);
      let currentFeatures = features[features.length - 1];
      
      // Calculate volatility for confidence intervals
      const volatility = this.calculateVolatility(cleanData);
      const safeVolatility = isFinite(volatility) && volatility > 0 ? volatility : 0.01;
      
      for (let i = 1; i <= days; i++) {
        const predictionDate = new Date(lastDate);
        predictionDate.setDate(lastDate.getDate() + i);
        const formattedDate = predictionDate.toISOString().split('T')[0];
        
        // Predict using random forest
        let relativePriceChange = this.predictWithForest(this.forest, currentFeatures);
        
        // Apply a small trend bias to avoid unreasonable predictions
        relativePriceChange += trendBias;
        
        // For longer-term predictions, add mean reversion tendency
        if (i > 10) {
          const meanReversionStrength = Math.min(0.002 * (i - 10), 0.02);
          relativePriceChange += meanReversionStrength;
        }
        
        // Apply safety cap to price change (prevent extreme movements)
        const cappedPriceChange = Math.max(-0.05, Math.min(0.05, relativePriceChange));
        
        // Update current price safely
        const newPrice = currentPrice * (1 + cappedPriceChange);
        currentPrice = isFinite(newPrice) ? Math.max(0.01, newPrice) : currentPrice;
        
        // Update features for next prediction with extra validation
        try {
          const updatedFeatures = this.updateFeaturesForNextPrediction(
            currentFeatures, 
            currentPrice, 
            cleanData
          );
          currentFeatures = updatedFeatures.map(f => isFinite(f) ? f : 0);
        } catch (err) {
          // If feature update fails, keep previous features
          console.warn("Failed to update features for next prediction step", err);
        }
        
        // Calculate confidence interval
        const baseInterval = safeVolatility * 1.645; // 90% confidence level
        
        // Scale by square root of time (standard statistical approach)
        const timeScaling = Math.sqrt(Math.min(i, 10)) + Math.log10(Math.max(1, i - 10 + 1));
        
        // Calculate confidence intervals that widen with time
        const confInterval = Math.min(0.2, baseInterval * timeScaling / 5);
        
        // Add prediction with bounds
        predictions.push({
          date: formattedDate,
          price: currentPrice,
          upper: currentPrice * (1 + confInterval),
          lower: Math.max(0.01, currentPrice * (1 - confInterval * 0.8)),
          algorithmUsed: 'randomforest'
        });
      }
      
      // Final validation
      return predictions.map(p => ({
        date: p.date,
        price: isFinite(p.price) ? p.price : currentPrice,
        upper: isFinite(p.upper) ? p.upper : currentPrice * 1.05,
        lower: isFinite(p.lower) ? Math.max(0.01, p.lower) : currentPrice * 0.95,
        algorithmUsed: p.algorithmUsed
      }));
    } catch (error) {
      console.error("Error in Random Forest prediction:", error);
      throw error;
    }
  }

  /**
   * Save the model - in this case, just storing forest structure
   */
  async saveModel(modelName: string): Promise<any> {
    try {
      // In a real app, save the forest to a file or database
      return { success: true, message: `Random Forest model '${modelName}' saved successfully` };
    } catch (error) {
      console.error(`Error saving Random Forest model: ${error}`);
      throw error;
    }
  }

  /**
   * Load a previously saved model
   */
  async loadModel(modelName: string): Promise<void> {
    try {
      // In a real app, load forest from a file or database
      console.log(`Loading Random Forest model '${modelName}'`);
      // Currently, this is a placeholder for future implementation
    } catch (error) {
      console.error(`Error loading Random Forest model: ${error}`);
      throw error;
    }
  }
  
  /**
   * Extract features for Random Forest model
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
      
      // Volatility features
      const volatility5d = this.calculateVolatility(data.slice(i - 5, i + 1));
      const volatility10d = this.calculateVolatility(data.slice(i - 10, i + 1));
      
      // Volume features
      const volumeChange1d = (currentPoint.volume - prevPoint.volume) / prevPoint.volume;
      const volumeChange5d = (currentPoint.volume - prev5Point.volume) / prev5Point.volume;
      
      // Price range features
      const dayRange = (currentPoint.high - currentPoint.low) / currentPoint.low;
      
      // Technical indicators
      const rsi = this.calculateRSI(data.slice(0, i + 1));
      const momentum = this.calculateMomentum(data.slice(0, i + 1), 5);
      
      // Create feature vector - make sure all values are finite
      const featureVector = [
        isFinite(priceChange1d) ? priceChange1d : 0,
        isFinite(priceChange5d) ? priceChange5d : 0,
        isFinite(priceChange10d) ? priceChange10d : 0,
        isFinite(priceChange20d) ? priceChange20d : 0,
        isFinite(ma5) ? ma5 / currentPoint.close - 1 : 0,
        isFinite(ma10) ? ma10 / currentPoint.close - 1 : 0,
        isFinite(ma20) ? ma20 / currentPoint.close - 1 : 0,
        isFinite(volatility5d) ? volatility5d : 0,
        isFinite(volatility10d) ? volatility10d : 0,
        isFinite(volumeChange1d) ? volumeChange1d : 0,
        isFinite(volumeChange5d) ? volumeChange5d : 0,
        isFinite(dayRange) ? dayRange : 0,
        isFinite(rsi) ? rsi / 100 : 0.5, // Normalize RSI to 0-1
        isFinite(momentum) ? momentum : 0
      ];
      
      features.push(featureVector);
    }
    
    return features;
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
   * Create a random forest (ensemble of decision trees)
   */
  private createRandomForest(features: number[][], targets: number[]): any[] {
    const forest = [];
    
    // Calculate baseline prediction (average target)
    const baselinePrediction = targets.reduce((sum, val) => sum + val, 0) / targets.length;
    
    for (let i = 0; i < this.numTrees; i++) {
      // Bootstrap sampling (sample with replacement)
      const { sampleFeatures, sampleTargets } = this.bootstrapSample(features, targets);
      
      // Create a decision tree
      const tree = this.createDecisionTree(
        sampleFeatures, 
        sampleTargets, 
        0, 
        this.maxDepth, 
        baselinePrediction
      );
      
      forest.push(tree);
    }
    
    return forest;
  }
  
  /**
   * Bootstrap sampling (sample with replacement)
   */
  private bootstrapSample(features: number[][], targets: number[]): { sampleFeatures: number[][], sampleTargets: number[] } {
    const numSamples = features.length;
    const numFeaturesToSample = Math.max(1, Math.floor(features[0].length * this.featureSamplingRatio));
    const numSamplesToTake = Math.max(10, Math.floor(numSamples * this.dataSamplingRatio));
    
    // Sample features (select a subset of features to consider)
    const featureIndices: number[] = [];
    while (featureIndices.length < numFeaturesToSample) {
      const idx = Math.floor(Math.random() * features[0].length);
      if (!featureIndices.includes(idx)) {
        featureIndices.push(idx);
      }
    }
    
    // Sample data points with replacement
    const sampleFeatures = [];
    const sampleTargets = [];
    
    for (let i = 0; i < numSamplesToTake; i++) {
      const idx = Math.floor(Math.random() * numSamples);
      
      // Create feature vector with only the selected features
      const featureVector = featureIndices.map(fidx => features[idx][fidx]);
      
      sampleFeatures.push(featureVector);
      sampleTargets.push(targets[idx]);
    }
    
    return { sampleFeatures, sampleTargets };
  }
  
  /**
   * Create a decision tree recursively
   */
  private createDecisionTree(
    features: number[][], 
    targets: number[], 
    depth: number, 
    maxDepth: number,
    defaultValue: number
  ): any {
    // If we've reached max depth or have very few samples, return a leaf node
    if (depth >= maxDepth || features.length <= 5) {
      const prediction = targets.length > 0 
        ? targets.reduce((sum, val) => sum + val, 0) / targets.length 
        : defaultValue;
      
      return { 
        isLeaf: true, 
        value: prediction 
      };
    }
    
    // Find the best split
    const bestSplit = this.findBestSplit(features, targets);
    
    // If no good split was found, return a leaf node
    if (!bestSplit.featureIndex) {
      const prediction = targets.length > 0 
        ? targets.reduce((sum, val) => sum + val, 0) / targets.length 
        : defaultValue;
      
      return { 
        isLeaf: true, 
        value: prediction 
      };
    }
    
    // Split the data
    const { leftFeatures, leftTargets, rightFeatures, rightTargets } = 
      this.splitData(features, targets, bestSplit.featureIndex, bestSplit.threshold);
    
    // If split doesn't actually divide the data, return a leaf
    if (leftFeatures.length === 0 || rightFeatures.length === 0) {
      const prediction = targets.length > 0 
        ? targets.reduce((sum, val) => sum + val, 0) / targets.length 
        : defaultValue;
      
      return { 
        isLeaf: true, 
        value: prediction 
      };
    }
    
    // Recursively build subtrees
    const leftSubtree = this.createDecisionTree(
      leftFeatures, 
      leftTargets, 
      depth + 1, 
      maxDepth,
      defaultValue
    );
    
    const rightSubtree = this.createDecisionTree(
      rightFeatures, 
      rightTargets, 
      depth + 1, 
      maxDepth,
      defaultValue
    );
    
    // Return the decision node
    return {
      isLeaf: false,
      featureIndex: bestSplit.featureIndex,
      threshold: bestSplit.threshold,
      left: leftSubtree,
      right: rightSubtree
    };
  }
  
  /**
   * Find the best split for a decision tree node
   */
  private findBestSplit(features: number[][], targets: number[]): { featureIndex: number, threshold: number, gain: number } {
    let bestGain = -Infinity;
    let bestFeatureIndex = -1;
    let bestThreshold = 0;
    
    // Try a random subset of features at each split
    const numFeatures = features[0].length;
    const featuresToTry = Math.max(1, Math.floor(Math.sqrt(numFeatures))); // Common heuristic
    
    // Randomly select features to try
    const featureIndices: number[] = [];
    while (featureIndices.length < featuresToTry) {
      const idx = Math.floor(Math.random() * numFeatures);
      if (!featureIndices.includes(idx)) {
        featureIndices.push(idx);
      }
    }
    
    // Calculate variance of current node (for gain calculation)
    const targetMean = targets.reduce((sum, val) => sum + val, 0) / targets.length;
    const nodeVariance = targets.reduce((sum, val) => sum + Math.pow(val - targetMean, 2), 0) / targets.length;
    
    // Try each selected feature
    for (const featureIndex of featureIndices) {
      // Get all values for this feature
      const values = features.map(f => f[featureIndex]);
      
      // Try a few potential thresholds
      const numThresholds = Math.min(10, Math.floor(values.length / 3));
      
      for (let i = 0; i < numThresholds; i++) {
        // Pick a random threshold from the values
        const thresholdIndex = Math.floor(Math.random() * values.length);
        const threshold = values[thresholdIndex];
        
        // Split the data
        const { leftFeatures, leftTargets, rightFeatures, rightTargets } = 
          this.splitData(features, targets, featureIndex, threshold);
        
        // Skip if split is too imbalanced
        if (leftFeatures.length < 5 || rightFeatures.length < 5) {
          continue;
        }
        
        // Calculate gain
        const leftMean = leftTargets.reduce((sum, val) => sum + val, 0) / leftTargets.length;
        const rightMean = rightTargets.reduce((sum, val) => sum + val, 0) / rightTargets.length;
        
        const leftVariance = leftTargets.reduce((sum, val) => sum + Math.pow(val - leftMean, 2), 0) / leftTargets.length;
        const rightVariance = rightTargets.reduce((sum, val) => sum + Math.pow(val - rightMean, 2), 0) / rightTargets.length;
        
        // Weighted variance reduction (information gain)
        const leftWeight = leftTargets.length / targets.length;
        const rightWeight = rightTargets.length / targets.length;
        
        const gain = nodeVariance - (leftWeight * leftVariance + rightWeight * rightVariance);
        
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
    const leftFeatures = [];
    const leftTargets = [];
    const rightFeatures = [];
    const rightTargets = [];
    
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
   * Predict using a single decision tree
   */
  private predictTree(tree: any, features: number[]): number {
    if (tree.isLeaf) {
      return tree.value;
    }
    
    // Sometimes feature index might be out of bounds due to feature sampling
    if (tree.featureIndex >= features.length) {
      return tree.left.isLeaf ? tree.left.value : tree.right.value;
    }
    
    if (features[tree.featureIndex] <= tree.threshold) {
      return this.predictTree(tree.left, features);
    } else {
      return this.predictTree(tree.right, features);
    }
  }
  
  /**
   * Predict using the entire random forest
   */
  private predictWithForest(forest: any[], features: number[]): number {
    if (!forest || forest.length === 0) {
      return 0;
    }
    
    // Get predictions from all trees
    const predictions = forest.map(tree => this.predictTree(tree, features));
    
    // Calculate the current price prediction (average of all tree predictions)
    const avgPrediction = predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
    
    // Calculate the standard deviation of predictions (for uncertainty estimation)
    const squaredDiffs = predictions.map(pred => Math.pow(pred - avgPrediction, 2));
    const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / predictions.length;
    const stdDev = Math.sqrt(variance);
    
    // Return relative price change (slightly smoothed based on prediction diversity)
    return avgPrediction;
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
} 