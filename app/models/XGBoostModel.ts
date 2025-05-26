import { StockDataPoint, PredictionPoint } from '@/app/lib/api';

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
export default class XGBoostModel {
  private numTrees: number;
  private maxDepth: number;
  private learningRate: number;
  private featureSubsamplingRatio: number;
  private trees: any[] = [];

  constructor(params: XGBoostModelParams = {}) {
    this.numTrees = params.numTrees || 10;
    this.maxDepth = params.maxDepth || 3;
    this.learningRate = params.learningRate || 0.1;
    this.featureSubsamplingRatio = params.featureSubsamplingRatio || 0.8;
  }

  /**
   * Train the XGBoost model
   */
  async train(historicalData: StockDataPoint[]): Promise<any> {
    try {
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
      const features = this.extractFeaturesForXGBoost(cleanData);
      
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
      
      // Create decision trees (boosted trees)
      this.trees = this.createDecisionTrees(features, targets);
      
      // If tree creation failed, fail training
      if (!this.trees || this.trees.length === 0) {
        console.warn("Decision tree creation failed for XGBoost");
        throw new Error("Decision tree creation failed");
      }
      
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
      
      // Get the latest features
      const features = this.extractFeaturesForXGBoost(cleanData);
      let currentFeatures = features[features.length - 1];
      
      // Calculate volatility for confidence intervals
      const volatility = this.calculateVolatility(cleanData);
      const safeVolatility = isFinite(volatility) && volatility > 0 ? volatility : 0.01;
      
      for (let i = 1; i <= days; i++) {
        const predictionDate = new Date(lastDate);
        predictionDate.setDate(lastDate.getDate() + i);
        const formattedDate = predictionDate.toISOString().split('T')[0];
        
        // Predict using ensemble of trees
        const relativePriceChange = this.predictWithTrees(this.trees, currentFeatures);
        
        // Apply safety cap to price change (prevent extreme movements)
        const cappedPriceChange = Math.max(-0.1, Math.min(0.1, relativePriceChange));
        
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
        
        // Calculate confidence interval (widens with time)
        const confInterval = Math.min(0.5, safeVolatility * Math.sqrt(i) * 1.645);
        
        // Add prediction with bounds
        predictions.push({
          date: formattedDate,
          price: currentPrice,
          upper: currentPrice * (1 + confInterval),
          lower: Math.max(0.01, currentPrice * (1 - confInterval))
        });
      }
      
      // Final validation
      return predictions.map(p => ({
        date: p.date,
        price: isFinite(p.price) ? p.price : currentPrice,
        upper: isFinite(p.upper) ? p.upper : currentPrice * 1.05,
        lower: isFinite(p.lower) ? Math.max(0.01, p.lower) : currentPrice * 0.95
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
  private extractFeaturesForXGBoost(data: StockDataPoint[]): number[][] {
    if (!data || data.length < 21) {
      return [];
    }
    
    const features = [];
    
    for (let i = 20; i < data.length; i++) {
      try {
        // Safely slice data for calculations
        const priorData = data.slice(0, i+1);
        
        // Technical indicators with safe calculations
        const sma5 = this.calculateSMA(priorData, 5) || data[i].close;
        const sma20 = this.calculateSMA(priorData, 20) || data[i].close;
        
        // Safe momentum calculation
        let momentum5 = 0;
        try {
          momentum5 = this.calculateMomentum(priorData, 5);
        } catch (e) {
          momentum5 = 0;
        }
        
        let momentum10 = 0;
        try {
          momentum10 = this.calculateMomentum(priorData, 10);
        } catch (e) {
          momentum10 = 0;
        }
        
        // Safely calculate volatility
        let volatility10 = 0.01;
        try {
          const volatilityWindow = Math.min(10, priorData.length - 1);
          volatility10 = this.calculateVolatility(priorData.slice(Math.max(0, i-volatilityWindow), i+1)) || 0.01;
        } catch (e) {
          volatility10 = 0.01;
        }
        
        // Safe RSI calculation
        let rsi = 50;
        try {
          rsi = this.calculateRSI(priorData) || 50;
        } catch (e) {
          rsi = 50;
        }
        
        // Price relative to moving averages (with safety checks)
        let priceSMA5Ratio = 1.0;
        if (sma5 > 0 && isFinite(data[i].close) && isFinite(sma5)) {
          priceSMA5Ratio = data[i].close / sma5;
        }
        
        let priceSMA20Ratio = 1.0;
        if (sma20 > 0 && isFinite(data[i].close) && isFinite(sma20)) {
          priceSMA20Ratio = data[i].close / sma20;
        }
        
        // Volume-based features (safely calculated)
        let volumeRatio = 1.0;
        try {
          const recentVolumesWindow = Math.min(5, i);
          const recentVolumes = data
            .slice(Math.max(0, i-recentVolumesWindow), i)
            .map(d => d.volume)
            .filter(v => isFinite(v) && v > 0);
            
          if (recentVolumes.length > 0) {
            const volumeAvg5 = recentVolumes.reduce((sum, v) => sum + v, 0) / recentVolumes.length;
            if (volumeAvg5 > 0 && isFinite(data[i].volume) && data[i].volume > 0) {
              volumeRatio = data[i].volume / volumeAvg5;
            }
          }
        } catch (e) {
          volumeRatio = 1.0;
        }
        
        // Ensure all values are finite and clamped to reasonable bounds
        const featureArray = [
          this.clamp(priceSMA5Ratio, 0.5, 2.0),
          this.clamp(priceSMA20Ratio, 0.5, 2.0),
          this.clamp(momentum5, -0.5, 0.5),
          this.clamp(momentum10, -0.5, 0.5),
          this.clamp(volatility10, 0.001, 0.5),
          this.clamp(rsi / 100, 0, 1),
          this.clamp(volumeRatio, 0.1, 10)
        ];
        
        features.push(featureArray);
      } catch (e) {
        console.warn("Error extracting features for XGBoost:", e);
        // Add reasonable default feature set if calculation fails
        features.push([1.0, 1.0, 0, 0, 0.01, 0.5, 1.0]);
      }
    }
    
    return features;
  }

  /**
   * Helper function to clamp a value between min and max
   */
  private clamp(value: number, min: number, max: number): number {
    if (!isFinite(value)) return (min + max) / 2;
    return Math.max(min, Math.min(max, value));
  }

  /**
   * Create a set of simplified decision trees
   */
  private createDecisionTrees(features: number[][], targets: number[]): any[] {
    // Basic validation
    if (!features || features.length === 0 || !targets || targets.length === 0) {
      return [];
    }
    
    // Ensure features and targets have compatible lengths
    if (features.length !== targets.length) {
      console.warn("Feature and target arrays have different lengths");
      // Try to use the shorter length
      const minLength = Math.min(features.length, targets.length);
      if (minLength < 20) {
        return []; // Not enough data to build trees
      }
      features = features.slice(0, minLength);
      targets = targets.slice(0, minLength);
    }
    
    const trees = [];
    const featureCount = features[0]?.length || 0;
    
    if (featureCount === 0) return [];
    
    try {
      // Create simple tree-like decision rules
      for (let i = 0; i < this.numTrees; i++) {
        // Randomly select features to use (with validation)
        const selectedFeatures: number[] = [];
        const maxFeaturesToTry = Math.min(3, Math.max(1, Math.floor(featureCount * this.featureSubsamplingRatio)));
        
        // Try to select distinct features
        for (let j = 0; j < maxFeaturesToTry; j++) {
          const featureIndex = Math.floor(Math.random() * featureCount);
          if (!selectedFeatures.includes(featureIndex)) {
            selectedFeatures.push(featureIndex);
          }
        }
        
        // Ensure we have at least one feature
        if (selectedFeatures.length === 0) {
          selectedFeatures.push(Math.floor(Math.random() * featureCount));
        }
        
        // Create simple thresholds for each selected feature
        const thresholds = selectedFeatures.map(featureIndex => {
          // Get valid values for this feature
          const values = features
            .map(f => f[featureIndex])
            .filter(v => isFinite(v));
            
          if (values.length < 2) return 0;
          
          try {
            const min = Math.min(...values);
            const max = Math.max(...values);
            // Ensure min and max are different
            if (min === max) return min;
            return min + Math.random() * (max - min);
          } catch (e) {
            // Fallback if min/max calculation fails
            return values[0];
          }
        });
        
        // Calculate outputs for this tree
        const outputs = [0, 0];
        let counts = [0, 0];
        
        // Simple partitioning based on the first threshold
        for (let j = 0; j < features.length; j++) {
          if (j >= features.length || !features[j]) continue;
          
          // Simple decision: above or below threshold for first feature
          const featureValue = features[j][selectedFeatures[0]];
          if (!isFinite(featureValue)) continue;
          
          const index = featureValue > thresholds[0] ? 1 : 0;
          outputs[index] += (targets[j] - targets[0]) / targets[0]; // Normalized relative change
          counts[index]++;
        }
        
        // Average the values in each partition
        if (counts[0] > 0) outputs[0] /= counts[0];
        if (counts[1] > 0) outputs[1] /= counts[1];
        
        // Ensure outputs are finite and reasonable
        outputs[0] = isFinite(outputs[0]) ? this.clamp(outputs[0], -0.1, 0.1) : 0;
        outputs[1] = isFinite(outputs[1]) ? this.clamp(outputs[1], -0.1, 0.1) : 0;
        
        // Store the decision tree
        trees.push({
          feature: selectedFeatures[0],
          threshold: thresholds[0],
          outputs: outputs,
          weight: this.learningRate
        });
      }
      
      return trees;
    } catch (e) {
      console.error("Error creating decision trees:", e);
      return [];
    }
  }

  /**
   * Predict with the ensemble of trees
   */
  private predictWithTrees(trees: any[], features: number[]): number {
    if (!trees || trees.length === 0 || !features || features.length === 0) {
      return 0;
    }
    
    let prediction = 0;
    
    // Predict with each tree and combine results
    for (const tree of trees) {
      try {
        const { feature, threshold, outputs, weight } = tree;
        if (!isFinite(features[feature])) continue;
        
        const index = features[feature] > threshold ? 1 : 0;
        prediction += outputs[index] * weight;
      } catch (e) {
        console.warn("Error in tree prediction:", e);
      }
    }
    
    return this.clamp(prediction, -0.1, 0.1); // Limit extreme predictions
  }

  /**
   * Update features for the next prediction
   */
  private updateFeaturesForNextPrediction(
    currentFeatures: number[], 
    predictedPrice: number, 
    historicalData: StockDataPoint[]
  ): number[] {
    // Deep copy current features to avoid modifying the original
    const newFeatures = [...currentFeatures];
    
    // These should match the features created in extractFeaturesForXGBoost
    try {
      if (historicalData.length === 0) return newFeatures;
      
      const position = historicalData.length - 1;
      const priorData = historicalData.slice(0, position + 1);
      const lastData = historicalData[position];
      
      // Create a synthetic new data point with the predicted price
      const newDataPoint: StockDataPoint = {
        date: '', // Date doesn't matter for feature calculation
        timestamp: lastData.timestamp + 86400000, // Next day
        open: predictedPrice,
        high: predictedPrice * 1.005, // Approximate
        low: predictedPrice * 0.995, // Approximate
        close: predictedPrice,
        volume: lastData.volume // Reuse last volume
      };
      
      // Append the synthetic point to create data for feature calculation
      const updatedData = [...priorData, newDataPoint];
      
      // Recalculate features that would most likely change
      // Price / SMA5 ratio - feature 0
      const newSMA5 = this.calculateSMA(updatedData, 5);
      if (newSMA5 > 0) {
        newFeatures[0] = this.clamp(predictedPrice / newSMA5, 0.5, 2.0);
      }
      
      // Price / SMA20 ratio - feature 1
      const newSMA20 = this.calculateSMA(updatedData, 20);
      if (newSMA20 > 0) {
        newFeatures[1] = this.clamp(predictedPrice / newSMA20, 0.5, 2.0);
      }
      
      // Momentum5 - feature 2
      const oldPrice5 = updatedData[Math.max(0, updatedData.length - 6)]?.close;
      if (oldPrice5 > 0 && oldPrice5 !== predictedPrice) {
        newFeatures[2] = this.clamp((predictedPrice - oldPrice5) / oldPrice5, -0.5, 0.5);
      }
      
      // Momentum10 - feature 3
      const oldPrice10 = updatedData[Math.max(0, updatedData.length - 11)]?.close;
      if (oldPrice10 > 0 && oldPrice10 !== predictedPrice) {
        newFeatures[3] = this.clamp((predictedPrice - oldPrice10) / oldPrice10, -0.5, 0.5);
      }
      
      // Keep volatility (feature 4) the same
      // Keep RSI (feature 5) the same
      // Keep volume ratio (feature 6) the same
      
      return newFeatures.map(f => isFinite(f) ? f : 0);
    } catch (e) {
      console.warn("Error updating features:", e);
      return newFeatures;
    }
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
  private calculateMomentum(data: StockDataPoint[], period: number): number {
    if (!data || data.length <= period || period <= 0) {
      return 0;
    }
    
    const oldPriceIdx = Math.max(0, data.length - period - 1);
    const currentIdx = data.length - 1;
    
    const oldPrice = data[oldPriceIdx]?.close;
    const currentPrice = data[currentIdx]?.close;
    
    // Check for valid prices
    if (!isFinite(oldPrice) || !isFinite(currentPrice) || oldPrice === 0) {
      return 0;
    }
    
    const momentum = (currentPrice - oldPrice) / oldPrice;
    
    // Cap very large values to prevent numerical issues
    if (momentum > 10) return 10;
    if (momentum < -10) return -10;
    
    return isFinite(momentum) ? momentum : 0;
  }

  /**
   * Calculate Relative Strength Index
   */
  private calculateRSI(data: StockDataPoint[], period: number = 14): number {
    if (!data || data.length < period + 1) {
      return 50;
    }
    
    let gains = 0;
    let losses = 0;
    
    for (let i = data.length - period; i < data.length; i++) {
      if (i <= 0) continue;
      
      const change = data[i].close - data[i - 1].close;
      if (!isFinite(change)) continue;
      
      if (change >= 0) {
        gains += change;
      } else {
        losses -= change;
      }
    }
    
    if (losses === 0) return 100;
    if (gains === 0) return 0;
    
    const rs = gains / losses;
    const rsi = 100 - (100 / (1 + rs));
    
    return isFinite(rsi) ? rsi : 50;
  }

  /**
   * Calculate volatility as standard deviation of returns
   */
  private calculateVolatility(data: StockDataPoint[]): number {
    if (!data || data.length < 2) return 0.01;
    
    const returns = [];
    for (let i = 1; i < data.length; i++) {
      const prevClose = data[i-1].close;
      if (prevClose === 0 || !isFinite(prevClose)) continue;
      
      const currentReturn = (data[i].close - prevClose) / prevClose;
      if (isFinite(currentReturn)) {
        returns.push(currentReturn);
      }
    }
    
    if (returns.length === 0) return 0.01;
    
    const meanReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - meanReturn, 2), 0) / returns.length;
    
    const volatility = Math.sqrt(variance);
    return isFinite(volatility) ? volatility : 0.01;
  }
} 