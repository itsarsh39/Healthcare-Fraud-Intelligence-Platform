Healthcare Fraud Intelligence Platform
The platform has been successfully built! This interactive system provides advanced capabilities to detect, investigate, and explain fraudulent healthcare claims using cutting-edge machine learning and network analysis.

What Was Built
The project is structured into three main capabilities:

1. Synthetic Data Generation (ml/generate_data.py)
We created a highly realistic, synthetic dataset mimicking a healthcare insurance environment.

Entities: Patients, Providers, and Claims.
Fraud Patterns Injected:
Upcoding: Billing for expensive procedures that don't match the patient profile.
Collusion Rings: A tight network of corrupt providers and "ghost" patients repeatedly billing low-value claims to avoid scrutiny.
2. Machine Learning Pipeline (ml/train_models.py)
We built a multi-layered detection system:

Graph Features (NetworkX): Extracted provider-patient degree centralities to quantify how connected they are, which is crucial for finding collusion.
Anomaly Detection (Isolation Forest): Identifies statistical outliers in claim patterns without needing historical labels.
Imbalanced Classification (XGBoost): Trained to predict the is_fraud flag using class weighting to handle the scarcity of fraudulent claims compared to normal ones.
Explainability (SHAP): Integrated into inference to explain the driving factors behind every high-risk prediction.
3. Interactive Dashboard (app.py)
A custom Streamlit application with a premium dark theme to visualize intelligence.

System Overview: High-level risk metrics, financial exposure, and real-time breakdowns of fraud by medical specialty.
Network Investigation: An interactive graph (using streamlit-agraph) to visually hunt for suspicious clusters of providers and patients sharing too many claims.
Claim Analysis: A deep-dive view into individual high-risk claims featuring a SHAP waterfall plot that explains exactly why the AI flagged the claim.
Verification
Local Testing
The model training pipeline successfully extracts network features and trains XGBoost and Isolation Forest. The Streamlit app renders the UI correctly and loads the models for dynamic inference.
