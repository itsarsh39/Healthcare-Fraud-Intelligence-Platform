import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import pickle
import shap

# Cache models so we don't load them on every call
_models = {}

def load_models():
    if not _models:
        print("Loading models...")
        _models['iso_forest'] = joblib.load('models/iso_forest.joblib')
        
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('models/xgboost_fraud.json')
        _models['xgb_model'] = xgb_model
        
        with open('models/feature_names.pkl', 'rb') as f:
            _models['feature_names'] = pickle.load(f)
            
        # Initialize SHAP explainer
        _models['explainer'] = shap.TreeExplainer(xgb_model)
        
    return _models

def predict_fraud(claim_data, models=None):
    """
    Predict fraud probability for a given claim.
    claim_data should be a DataFrame with a single row or multiple rows.
    """
    if models is None:
        models = load_models()
        
    # Ensure all features are present in the correct order
    feature_names = models['feature_names']
    
    # Fill missing columns with 0 (e.g., if a specific specialty wasn't in this sample)
    for col in feature_names:
        if col not in claim_data.columns:
            claim_data[col] = 0
            
    X = claim_data[feature_names]
    
    # Calculate anomaly score if not present
    if 'anomaly_score' not in X.columns:
        # We need the base features without anomaly score to calculate it
        # Actually, in our training setup, we calculated anomaly score BEFORE classification
        # Let's assume the input already has anomaly score for simplicity, or we compute it
        base_features = [c for c in feature_names if c != 'anomaly_score']
        X_base = X[base_features]
        X['anomaly_score'] = models['iso_forest'].decision_function(X_base)
    
    # Predict
    probabilities = models['xgb_model'].predict_proba(X)[:, 1]
    return probabilities

def explain_prediction(claim_data, models=None):
    """
    Generate SHAP values for a given claim.
    """
    if models is None:
        models = load_models()
        
    feature_names = models['feature_names']
    
    for col in feature_names:
        if col not in claim_data.columns:
            claim_data[col] = 0
            
    X = claim_data[feature_names]
    
    shap_values = models['explainer'](X)
    return shap_values
