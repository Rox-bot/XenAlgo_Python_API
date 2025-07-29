from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import json

app = FastAPI(
    title="Trading Psychology AI",
    description="AI-powered trading psychology analysis and behavioral pattern detection",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class Trade(BaseModel):
    id: str
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_date: str
    exit_date: Optional[str] = None
    pnl: Optional[float] = None
    emotion_before: Optional[str] = None
    emotion_after: Optional[str] = None
    reasoning: Optional[str] = None

class UserBehaviorData(BaseModel):
    user_id: str
    trades: List[Trade]
    risk_tolerance: float  # 1-10 scale
    trading_experience: str  # BEGINNER/INTERMEDIATE/ADVANCED
    capital_amount: float
    goals: List[str]  # ["INCOME", "GROWTH", "PRESERVATION"]

class BehavioralAnalysis(BaseModel):
    user_id: str
    risk_score: float
    behavioral_patterns: List[str]
    emotional_trends: Dict[str, float]
    performance_correlation: Dict[str, float]
    recommendations: List[str]
    risk_level: str  # LOW/MEDIUM/HIGH
    confidence_score: float

# Global variables for ML models
risk_model = None
pattern_detector = None
scaler = StandardScaler()

def load_models():
    """Load pre-trained ML models"""
    global risk_model, pattern_detector
    
    # In production, load from cloud storage
    # For now, create simple models
    risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
    pattern_detector = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Train with dummy data for now
    # In production, train with real trading data
    X_dummy = np.random.rand(1000, 10)
    y_dummy = np.random.randint(0, 3, 1000)
    
    risk_model.fit(X_dummy, y_dummy)
    pattern_detector.fit(X_dummy, y_dummy)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "trading-psychology-ai",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": risk_model is not None
    }

@app.post("/analyze-behavior", response_model=BehavioralAnalysis)
async def analyze_trading_behavior(data: UserBehaviorData):
    """Analyze user's trading behavior and provide AI insights"""
    try:
        # Extract features from trades
        features = extract_trading_features(data.trades)
        
        # Calculate risk score
        risk_score = calculate_risk_score(features, data.risk_tolerance)
        
        # Detect behavioral patterns
        patterns = detect_behavioral_patterns(data.trades)
        
        # Analyze emotional trends
        emotional_trends = analyze_emotional_trends(data.trades)
        
        # Calculate performance correlation
        performance_correlation = calculate_performance_correlation(data.trades)
        
        # Generate recommendations
        recommendations = generate_recommendations(
            data.risk_tolerance, 
            patterns, 
            risk_score,
            data.trading_experience
        )
        
        # Determine risk level
        risk_level = determine_risk_level(risk_score)
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(len(data.trades))
        
        return BehavioralAnalysis(
            user_id=data.user_id,
            risk_score=risk_score,
            behavioral_patterns=patterns,
            emotional_trends=emotional_trends,
            performance_correlation=performance_correlation,
            recommendations=recommendations,
            risk_level=risk_level,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def extract_trading_features(trades: List[Trade]) -> np.ndarray:
    """Extract numerical features from trading data"""
    if not trades:
        return np.zeros((1, 10))
    
    features = []
    for trade in trades:
        # Basic trading features
        trade_features = [
            trade.quantity,
            trade.entry_price,
            trade.pnl or 0,
            len(trade.reasoning or ""),  # Reasoning complexity
            # Add more features as needed
        ]
        features.append(trade_features)
    
    # Pad with zeros if needed
    while len(features) < 10:
        features.append([0] * len(features[0]) if features else [0] * 5)
    
    return np.array(features[:10])  # Take first 10 trades

def calculate_risk_score(features: np.ndarray, risk_tolerance: float) -> float:
    """Calculate risk score based on trading patterns"""
    if features.size == 0:
        return 0.5
    
    # Simple risk calculation (replace with ML model in production)
    avg_position_size = np.mean(features[:, 0]) if features.size > 0 else 0
    avg_pnl = np.mean(features[:, 2]) if features.size > 0 else 0
    
    # Normalize risk score between 0 and 1
    position_risk = min(avg_position_size / 1000, 1.0)  # Normalize by 1000
    pnl_risk = abs(avg_pnl) / 1000  # Normalize by 1000
    
    # Combine factors
    risk_score = (position_risk * 0.4 + pnl_risk * 0.3 + risk_tolerance * 0.1) / 10
    
    return max(0, min(1, risk_score))

def detect_behavioral_patterns(trades: List[Trade]) -> List[str]:
    """Detect common behavioral patterns"""
    patterns = []
    
    if len(trades) < 3:
        return patterns
    
    # Detect revenge trading
    consecutive_losses = 0
    for trade in trades[-5:]:  # Check last 5 trades
        if trade.pnl and trade.pnl < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
    
    if consecutive_losses >= 3:
        patterns.append("revenge_trading")
    
    # Detect FOMO trading
    recent_trades = trades[-3:]
    if len(recent_trades) >= 2:
        time_gaps = []
        for i in range(1, len(recent_trades)):
            trade1 = datetime.fromisoformat(recent_trades[i-1].entry_date)
            trade2 = datetime.fromisoformat(recent_trades[i].entry_date)
            time_gaps.append((trade2 - trade1).total_seconds() / 3600)  # Hours
        
        if any(gap < 1 for gap in time_gaps):  # Trades within 1 hour
            patterns.append("fomo_trading")
    
    # Detect overconfidence
    if len(trades) >= 5:
        recent_pnl = [t.pnl for t in trades[-5:] if t.pnl]
        if recent_pnl and all(pnl > 0 for pnl in recent_pnl):
            patterns.append("overconfidence")
    
    # Detect analysis paralysis
    reasoning_lengths = [len(t.reasoning or "") for t in trades[-5:]]
    if reasoning_lengths and np.mean(reasoning_lengths) > 200:  # Long reasoning
        patterns.append("analysis_paralysis")
    
    return patterns

def analyze_emotional_trends(trades: List[Trade]) -> Dict[str, float]:
    """Analyze emotional patterns in trading"""
    emotions = {
        "confident": 0,
        "anxious": 0,
        "excited": 0,
        "fearful": 0,
        "neutral": 0
    }
    
    total_trades = len(trades)
    if total_trades == 0:
        return emotions
    
    for trade in trades:
        emotion = trade.emotion_before or "neutral"
        emotions[emotion.lower()] += 1
    
    # Convert to percentages
    return {emotion: count / total_trades for emotion, count in emotions.items()}

def calculate_performance_correlation(trades: List[Trade]) -> Dict[str, float]:
    """Calculate correlation between emotions and performance"""
    if len(trades) < 3:
        return {"confident": 0.5, "anxious": 0.5}
    
    emotion_performance = {}
    
    for emotion in ["confident", "anxious", "excited", "fearful"]:
        emotion_trades = [t for t in trades if t.emotion_before == emotion and t.pnl]
        if emotion_trades:
            avg_pnl = np.mean([t.pnl for t in emotion_trades])
            emotion_performance[emotion] = avg_pnl
        else:
            emotion_performance[emotion] = 0
    
    return emotion_performance

def generate_recommendations(
    risk_tolerance: float, 
    patterns: List[str], 
    risk_score: float,
    experience: str
) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []
    
    # Risk-based recommendations
    if risk_score > 0.7:
        recommendations.append("Consider reducing position sizes by 50%")
        recommendations.append("Implement strict stop-loss orders")
    
    if risk_score < 0.3:
        recommendations.append("You can increase position sizes gradually")
        recommendations.append("Consider more aggressive strategies")
    
    # Pattern-based recommendations
    if "revenge_trading" in patterns:
        recommendations.append("Take a 30-minute break after 2 consecutive losses")
        recommendations.append("Implement a maximum daily loss limit")
    
    if "fomo_trading" in patterns:
        recommendations.append("Wait 15 minutes before entering new positions")
        recommendations.append("Set price alerts instead of immediate action")
    
    if "overconfidence" in patterns:
        recommendations.append("Review your risk management rules")
        recommendations.append("Consider taking partial profits")
    
    if "analysis_paralysis" in patterns:
        recommendations.append("Set time limits for trade analysis")
        recommendations.append("Use predefined entry/exit criteria")
    
    # Experience-based recommendations
    if experience == "BEGINNER":
        recommendations.append("Focus on paper trading for 3 months")
        recommendations.append("Start with small position sizes")
    
    # General recommendations
    recommendations.append("Keep a trading journal with emotions")
    recommendations.append("Review your trades weekly")
    
    return recommendations[:5]  # Return top 5 recommendations

def determine_risk_level(risk_score: float) -> str:
    """Determine risk level based on score"""
    if risk_score < 0.3:
        return "LOW"
    elif risk_score < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

def calculate_confidence_score(trade_count: int) -> float:
    """Calculate confidence in analysis based on data quality"""
    if trade_count < 5:
        return 0.3
    elif trade_count < 20:
        return 0.6
    else:
        return 0.9

@app.post("/predict-behavior")
async def predict_future_behavior(data: UserBehaviorData):
    """Predict future behavioral patterns"""
    try:
        # Extract features
        features = extract_trading_features(data.trades)
        
        # Use ML model to predict (simplified for now)
        prediction = risk_model.predict(features.reshape(1, -1))[0] if risk_model else 1
        
        return {
            "user_id": data.user_id,
            "predicted_risk_level": ["LOW", "MEDIUM", "HIGH"][prediction],
            "confidence": 0.75,
            "next_week_prediction": "Stable behavior expected"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "risk_model_loaded": risk_model is not None,
        "pattern_detector_loaded": pattern_detector is not None,
        "model_version": "1.0.0",
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 