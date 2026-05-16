import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import xgboost as xgb
import pickle
import os
import joblib

def load_data(filepath='data/healthcare_claims.csv'):
    df = pd.read_csv(filepath)
    return df

def extract_network_features(df):
    print("Extracting NetworkX graph features...")
    # Create a bipartite graph of patients and providers
    G = nx.Graph()
    
    # Add nodes and edges
    for _, row in df.iterrows():
        patient = row['patient_id']
        provider = row['provider_id']
        G.add_edge(patient, provider)
        
    # Calculate degree centrality
    centrality = nx.degree_centrality(G)
    
    # Map back to dataframe
    df['patient_degree'] = df['patient_id'].map(centrality)
    df['provider_degree'] = df['provider_id'].map(centrality)
    
    return df

def preprocess_features(df):
    print("Preprocessing features...")
    # Create derived features
    df['is_same_region'] = (df['region_patient'] == df['region_provider']).astype(int)
    
    # Select features for modeling
    feature_cols = ['claim_amount', 'age', 'patient_degree', 'provider_degree', 'is_same_region']
    
    # One-hot encode categorical features
    cat_cols = ['gender', 'specialty']
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)
    
    # Combine
    X = pd.concat([df[feature_cols], df_encoded], axis=1)
    y = df['is_fraud'].astype(int)
    
    return X, y

def train_anomaly_detection(X):
    print("Training Isolation Forest for Anomaly Detection...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso_forest.fit(X)
    
    # -1 for anomalies, 1 for normal. We map to an anomaly score (lower is more anomalous in sklearn, but let's just use the raw score)
    anomaly_scores = iso_forest.decision_function(X)
    
    # Add anomaly score as a feature for the classification model
    X_with_anomaly = X.copy()
    X_with_anomaly['anomaly_score'] = anomaly_scores
    
    return iso_forest, X_with_anomaly

def train_classification_model(X, y):
    print("Training XGBoost Classifier...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate scale_pos_weight for imbalanced dataset
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print("PR AUC Score:", average_precision_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model

def save_models(iso_forest, xgb_model, feature_names):
    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(iso_forest, 'models/iso_forest.joblib')
    xgb_model.save_model('models/xgboost_fraud.json')
    
    # Save feature names to ensure alignment during inference
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

def main():
    df = load_data()
    df = extract_network_features(df)
    X, y = preprocess_features(df)
    
    iso_forest, X_with_anomaly = train_anomaly_detection(X)
    xgb_model = train_classification_model(X_with_anomaly, y)
    
    save_models(iso_forest, xgb_model, X_with_anomaly.columns.tolist())
    print("Training complete.")

if __name__ == '__main__':
    main()
