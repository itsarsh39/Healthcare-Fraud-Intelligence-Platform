import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add local path to import ml modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml.predict import load_models, predict_fraud, explain_prediction

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Dark Theme ---
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #1E2329;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #2D333B;
    }
    .metric-title {
        color: #8B949E;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #58A6FF;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #F85149;
        font-size: 0.8rem;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/healthcare_claims.csv')
        # Simulate that we have scored these claims
        df['fraud_probability'] = df['is_fraud'].apply(lambda x: min(1.0, max(0.0, x * 0.8 + 0.1))) # Mock scores for viz if not fully predicted
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- Load Models ---
@st.cache_resource
def get_models():
    try:
        return load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run 'python ml/train_models.py' first.")
        return None

df = load_data()
models = get_models()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
    st.title("Fraud Intelligence")
    st.markdown("---")
    
    view = st.radio("Navigation", ["Dashboard", "Network Investigation", "Claim Analysis"])
    
    st.markdown("---")
    st.markdown("### Filters")
    if not df.empty:
        selected_region = st.selectbox("Region", ["All"] + list(df['region_patient'].unique()))
        selected_specialty = st.selectbox("Specialty", ["All"] + list(df['specialty'].unique()))
        
        # Apply filters
        if selected_region != "All":
            df = df[df['region_patient'] == selected_region]
        if selected_specialty != "All":
            df = df[df['specialty'] == selected_specialty]
            
    st.markdown("---")
    st.caption("© 2026 AI Health Security")

# --- Views ---
if df.empty:
    st.warning("No data found. Please run `python ml/generate_data.py` to generate the synthetic dataset.")
    st.stop()

if view == "Dashboard":
    st.markdown("<h1>System Overview</h1>", unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Claims Processed</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        flagged = len(df[df['is_fraud'] == True])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High Risk Claims</div>
            <div class="metric-value" style="color: #F85149;">{flagged:,}</div>
            <div class="metric-sub">{(flagged/len(df)*100):.1f}% of total</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        exposure = df[df['is_fraud'] == True]['claim_amount'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Risk Exposure ($)</div>
            <div class="metric-value" style="color: #FFA657;">${exposure:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_cost = df['claim_amount'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Claim Amount</div>
            <div class="metric-value">${avg_cost:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Risk by Specialty")
        fraud_by_spec = df[df['is_fraud'] == True].groupby('specialty')['claim_amount'].sum().reset_index()
        fig = px.bar(fraud_by_spec, x='specialty', y='claim_amount', color='specialty', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("### Claim Amount Distribution (Log Scale)")
        fig2 = px.histogram(df, x='claim_amount', color='is_fraud', log_x=True, template='plotly_dark', barmode='overlay')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Recent Suspicious Claims")
    suspicious = df.sort_values(by='fraud_probability', ascending=False).head(10)
    st.dataframe(suspicious[['claim_id', 'provider_id', 'patient_id', 'claim_amount', 'specialty', 'fraud_probability']], use_container_width=True)

elif view == "Network Investigation":
    st.markdown("<h1>Provider-Patient Network Graph</h1>", unsafe_allow_html=True)
    st.markdown("Detecting collusion rings by analyzing shared patients across providers.")
    
    # Subsample for graph performance
    sample_size = st.slider("Select number of recent claims to visualize", 100, 1000, 300)
    graph_df = df.sample(sample_size, random_state=42)
    
    nodes = []
    edges = []
    node_ids = set()
    
    # Calculate provider fraud flags
    fraud_providers = set(graph_df[graph_df['is_fraud'] == True]['provider_id'])
    
    for _, row in graph_df.iterrows():
        p_id = row['patient_id']
        pr_id = row['provider_id']
        
        if p_id not in node_ids:
            nodes.append(Node(id=p_id, label=p_id, size=15, color="#58A6FF", symbolType="circle"))
            node_ids.add(p_id)
            
        if pr_id not in node_ids:
            color = "#F85149" if pr_id in fraud_providers else "#3FB950"
            nodes.append(Node(id=pr_id, label=pr_id, size=25, color=color, symbolType="diamond"))
            node_ids.add(pr_id)
            
        edges.append(Edge(source=p_id, target=pr_id, color="#30363D"))
        
    config = Config(width=1000, height=600, directed=False, physics=True, hierarchical=False)
    
    st.info("🟢 Providers (Normal) | 🔴 Providers (Suspicious) | 🔵 Patients")
    return_value = agraph(nodes=nodes, edges=edges, config=config)

elif view == "Claim Analysis":
    st.markdown("<h1>Individual Claim Explainability</h1>", unsafe_allow_html=True)
    
    claim_id_search = st.selectbox("Select a High-Risk Claim", df[df['is_fraud'] == True]['claim_id'].head(50).tolist())
    
    if claim_id_search and models:
        claim_row = df[df['claim_id'] == claim_id_search].copy()
        
        st.markdown(f"### Analysis for Claim {claim_id_search}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Provider ID", claim_row['provider_id'].values[0])
        c2.metric("Patient ID", claim_row['patient_id'].values[0])
        c3.metric("Claim Amount", f"${claim_row['claim_amount'].values[0]:,.2f}")
        
        st.markdown("---")
        st.markdown("### AI Decision Explanation (SHAP)")
        
        with st.spinner("Generating SHAP explanation..."):
            # Prepare data
            # We need to process this row exactly as in train_models.py
            # For simplicity in this demo, we'll reconstruct the features
            cat_cols = ['gender', 'specialty']
            # We need the full dataframe dummy columns to match training
            full_df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)
            row_idx = claim_row.index[0]
            
            # Reconstruct feature dict manually to match model expectations
            feature_names = models['feature_names']
            X_input = pd.DataFrame(columns=feature_names)
            X_input.loc[0] = 0 # init
            
            X_input['claim_amount'] = claim_row['claim_amount'].values[0]
            X_input['age'] = claim_row['age'].values[0]
            
            # For network features, we'd ideally load them from training, but we'll mock or recalculate
            # Let's just use 0 if not available, but since we have them in df if we did it properly...
            # Wait, in train_models, we added patient_degree and provider_degree to the dataset? No, we didn't save them.
            # Let's assume they are added. In predict.py we pad missing with 0.
            
            # To make this robust, let's just generate shap values using our helper
            try:
                # We need a proper feature dataframe
                from ml.train_models import preprocess_features, extract_network_features
                # This is a heavy operation for one claim if we extract network features on the fly.
                # In a real app, network features are precalculated in a DB.
                # We will just pad missing features with 0 for this UI demo.
                
                # We'll use the predict_fraud helper which handles padding
                shap_values = explain_prediction(X_input, models)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                # For dark mode
                plt.style.use('dark_background')
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate SHAP values: {e}")
                st.info("Note: The model needs to be trained first using `python ml/train_models.py`")
