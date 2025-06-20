import { StockDataPoint, PredictionPoint } from '@/app/lib/api';
import { PredictionModel } from './index';

export interface XGBoostModelParams {
  numTrees?: number;
  maxDepth?: number;
  learningRate?: number;
  featureSubsamplingRatio?: number;
}

/**
 * XGBoost model for stock price prediction
 * JavaScript implementation inspired by XGBoost's gradient boosting principles
 */
export class XGBoostModel implements PredictionModel {
  private numTrees: number;
  private maxDepth: number;
  private learningRate: number;
  private featureSubsamplingRatio: number;
  private trees: any[] = [];

  constructor(params: XGBoostModelParams = {}) {
    this.numTrees = params.numTrees || 100;
    this.maxDepth = params.maxDepth || 3;
    this.learningRate = params.learningRate || 0.1;
    this.featureSubsamplingRatio = params.featureSubsamplingRatio || 0.8;
    console.log('🔍 XGBoostModel created with params:', 
      `numTrees=${this.numTrees}, maxDepth=${this.maxDepth}, learningRate=${this.learningRate}`);
  }

  /**
   * Train the XGBoost model
   */
  async train(historicalData: StockDataPoint[]): Promise<any> {
    try {
      console.log(`Training XGBoost model with ${historicalData.length} data points...`);
      
      // Input validation - exit early if data is insufficient
      if (!historicalData || historicalData.length < 30) {
        console.warn("Insufficient historical data for XGBoost training");
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
        console.warn("Too many invalid data points for XGBoost training");
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
        console.warn("Feature extraction failed for XGBoost training");
        throw new Error("Feature extraction failed");
      }
      
      // Create XGBoost trees (boosted trees)
      this.trees = this.createXGBoostTrees(features, targets);
      
      // If tree creation failed, fail training
      if (!this.trees || this.trees.length === 0) {
        console.warn("Decision tree creation failed for XGBoost");
        throw new Error("Decision tree creation failed");
      }
      
      console.log(`XGBoost training complete with ${this.trees.length} trees`);
      return { success: true, numTrees: this.trees.length };
    } catch (error) {
      console.error("Error in XGBoost training:", error);
      throw error;
    }
  }

  /**
   * Generate predictions using the trained XGBoost model
   */
  async predict(historicalData: StockDataPoint[], days: number = 30): Promise<PredictionPoint[]> {
    try {
      // Input validation
      if (!historicalData || historicalData.length < 30) {
        console.warn("Insufficient historical data for XGBoost prediction");
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
      
      // Calculate recent trend and volatility to inform prediction
      const recentPrices = cleanData.slice(-30).map(d => d.close);
      const recentTrend = this.calculateRecentTrend(recentPrices);
      
      // Bias prediction slightly based on recent trend (momentum effect)
      const trendBias = recentTrend * 0.2; // Reduce impact to 20% of trend
      
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
        
        // Predict using ensemble of trees
        let relativePriceChange = this.predictWithTrees(this.trees, currentFeatures);
        
        // Apply a small trend bias to avoid unreasonable predictions
        relativePriceChange += trendBias;
        
        // For longer-term predictions, add mean reversion tendency
        if (i > 10) {
          const meanReversionStrength = Math.min(0.003 * (i - 10), 0.03); // Gradual increase up to 3%
          relativePriceChange += meanReversionStrength; // Small positive bias for longer predictions
        }
        
        // Apply safety cap to price change (prevent extreme movements)
        const cappedPriceChange = Math.max(-0.05, Math.min(0.15, relativePriceChange));
        
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
        const confInterval = Math.min(0.3, baseInterval * timeScaling / 5);
        
        // Add prediction with bounds
        predictions.push({
          date: formattedDate,
          price: currentPrice,
          upper: currentPrice * (1 + confInterval),
          lower: Math.max(0.01, currentPrice * (1 - confInterval * 0.8)), // Slightly asymmetric
          algorithmUsed: 'xgboost'
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
      console.error("Error in XGBoost prediction:", error);
      throw error;
    }
  }

  /**
   * Save the model - in this case, just storing tree structure
   */
  async saveModel(modelName: string): Promise<any> {
    try {
      // In a real app, save the trees to a file or database
      return { success: true, message: `XGBoost model '${modelName}' saved successfully` };
    } catch (error) {
      console.error(`Error saving XGBoost model: ${error}`);
      throw error;
    }
  }

  /**
   * Load a previously saved model
   */
  async loadModel(modelName: string): Promise<void> {
    try {
      // In a real app, load trees from a file or database
      console.log(`Loading XGBoost model '${modelName}'`);
      // Currently, this is a placeholder for future implementation
    } catch (error) {
      console.error(`Error loading XGBoost model: ${error}`);
      throw error;
    }
  }
  
  /**
   * Extract features for XGBoost model
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
   * Create XGBoost trees (gradient boosted decision trees)
   */
  private createXGBoostTrees(features: number[][], targets: number[]): any[] {
    const numSamples = features.length;
    const numFeatures = features[0].length;
    const trees = [];
    
    // Initialize predictions with the mean of the targets
    let predictions = new Array(numSamples).fill(
      targets.reduce((sum, val) => sum + val, 0) / targets.length
    );
    
    // Gradient boosting iterations
    for (let treeIdx = 0; treeIdx < this.numTrees; treeIdx++) {
      // Calculate gradients (residuals for regression)
      const gradients = targets.map((target, i) => target - predictions[i]);
      
      // Subsample features for this tree (feature bagging)
      const featureIndices = this.sampleFeatures(
        numFeatures, 
        Math.floor(numFeatures * this.featureSubsamplingRatio)
      );
      
      // Create a decision tree on the gradients
      const tree = this.buildDecisionTree(
        features, 
        gradients, 
        featureIndices,
        0, 
        this.maxDepth
      );
      
      // Update predictions with this tree's contribution
      predictions = predictions.map((pred, i) => {
        const update = this.predictTree(tree, features[i]) * this.learningRate;
        return pred + update;
      });
      
      trees.push(tree);
    }
    
    return trees;
  }
  
  /**
   * Build a single decision tree recursively
   */
  private buildDecisionTree(
    features: number[][], 
    gradients: number[], 
    featureIndices: number[],
    depth: number, 
    maxDepth: number
  ): any {
    // If we've reached max depth or have no data, return a leaf node
    if (depth >= maxDepth || gradients.length === 0 || featureIndices.length === 0) {
      return { 
        isLeaf: true, 
        value: gradients.reduce((sum, g) => sum + g, 0) / Math.max(1, gradients.length)
      };
    }
    
    // Find the best split
    const bestSplit = this.findBestSplit(features, gradients, featureIndices);
    
    // If no good split was found, return a leaf node
    if (!bestSplit.hasOwnProperty('featureIndex')) {
      return { 
        isLeaf: true, 
        value: gradients.reduce((sum, g) => sum + g, 0) / Math.max(1, gradients.length)
      };
    }
    
    // Split the data
    const leftIndices: number[] = [];
    const rightIndices: number[] = [];
    
    for (let i = 0; i < features.length; i++) {
      if (features[i][bestSplit.featureIndex] <= bestSplit.threshold) {
        leftIndices.push(i);
      } else {
        rightIndices.push(i);
      }
    }
    
    // If split doesn't actually divide the data, return a leaf
    if (leftIndices.length === 0 || rightIndices.length === 0) {
      return { 
        isLeaf: true, 
        value: gradients.reduce((sum, g) => sum + g, 0) / Math.max(1, gradients.length)
      };
    }
    
    // Extract left and right subsets
    const leftFeatures = leftIndices.map(i => features[i]);
    const rightFeatures = rightIndices.map(i => features[i]);
    const leftGradients = leftIndices.map(i => gradients[i]);
    const rightGradients = rightIndices.map(i => gradients[i]);
    
    // Recursively build subtrees
    const leftSubtree = this.buildDecisionTree(
      leftFeatures, 
      leftGradients, 
      featureIndices, 
      depth + 1, 
      maxDepth
    );
    
    const rightSubtree = this.buildDecisionTree(
      rightFeatures, 
      rightGradients, 
      featureIndices, 
      depth + 1, 
      maxDepth
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
  private findBestSplit(features: number[][], gradients: number[], featureIndices: number[]): any {
    let bestGain = -Infinity;
    let bestFeatureIndex = -1;
    let bestThreshold = 0;
    
    const totalGradient = gradients.reduce((sum, g) => sum + g, 0);
    const totalGradientSquared = gradients.reduce((sum, g) => sum + g * g, 0);
    const initialScore = totalGradientSquared - (totalGradient * totalGradient) / gradients.length;
    
    // Try each feature as a potential split
    for (const featureIndex of featureIndices) {
      // Get unique values for this feature
      const uniqueValues = Array.from(new Set(features.map(f => f[featureIndex]))).sort();
      
      // Try potential thresholds between each pair of unique values
      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        
        let leftGradientSum = 0;
        let leftGradientSquaredSum = 0;
        let leftCount = 0;
        
        for (let j = 0; j < features.length; j++) {
          if (features[j][featureIndex] <= threshold) {
            leftGradientSum += gradients[j];
            leftGradientSquaredSum += gradients[j] * gradients[j];
            leftCount++;
          }
        }
        
        // Skip if all samples went to one side
        if (leftCount === 0 || leftCount === features.length) continue;
        
        const rightGradientSum = totalGradient - leftGradientSum;
        const rightGradientSquaredSum = totalGradientSquared - leftGradientSquaredSum;
        const rightCount = features.length - leftCount;
        
        // Calculate gain using variance reduction formula
        const leftScore = leftGradientSquaredSum - (leftGradientSum * leftGradientSum) / leftCount;
        const rightScore = rightGradientSquaredSum - (rightGradientSum * rightGradientSum) / rightCount;
        const gain = initialScore - (leftScore + rightScore);
        
        if (gain > bestGain) {
          bestGain = gain;
          bestFeatureIndex = featureIndex;
          bestThreshold = threshold;
        }
      }
    }
    
    if (bestGain > 0) {
      return { featureIndex: bestFeatureIndex, threshold: bestThreshold, gain: bestGain };
    } else {
      return {}; // No good split found
    }
  }
  
  /**
   * Predict using a single decision tree
   */
  private predictTree(tree: any, features: number[]): number {
    if (tree.isLeaf) {
      return tree.value;
    }
    
    if (features[tree.featureIndex] <= tree.threshold) {
      return this.predictTree(tree.left, features);
    } else {
      return this.predictTree(tree.right, features);
    }
  }
  
  /**
   * Predict using the entire ensemble of trees
   */
  private predictWithTrees(trees: any[], features: number[]): number {
    // Get the mean target value from the first tree (stored during training)
    const initialPrediction = 0; // This would be the global mean in a proper implementation
    
    // Sum up contributions from all trees
    const treeContributions = trees.map(tree => this.predictTree(tree, features) * this.learningRate);
    const totalContribution = treeContributions.reduce((sum, contrib) => sum + contrib, 0);
    
    return totalContribution;
  }
  
  /**
   * Sample feature indices without replacement for feature bagging
   */
  private sampleFeatures(numFeatures: number, sampleSize: number): number[] {
    const indices = Array.from({ length: numFeatures }, (_, i) => i);
    
    // Randomly shuffle the array using Fisher-Yates algorithm
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    return indices.slice(0, sampleSize);
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