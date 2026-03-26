import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_imbalanced_dataset, get_dataset_info, prepare_data
from utils.models import ModelEvaluator
from utils.smote_handler import SMOTEHandler
from utils.model_loader import get_model_loader
import warnings

warnings.filterwarnings('ignore')

# Set page config FIRST (before any other Streamlit calls)
st.set_page_config(
    page_title="SMOTE Virtual Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize model loader
@st.cache_resource
def init_model_loader():
    """Initialize the model loader."""
    return get_model_loader(model_dir="models")

model_loader = init_model_loader()

# Helper to make DataFrames hashable for caching
def hash_dataframe(df):
    return pd.util.hash_pandas_object(df, index=True).values

def hash_series(s):
    return pd.util.hash_pandas_object(s, index=True).values

# Caching functions to prevent unnecessary recomputation
@st.cache_data(hash_funcs={pd.DataFrame: hash_dataframe, pd.Series: hash_series})
def load_and_prep_data(dataset_name):
    """Load and prepare dataset with caching."""
    X, y = load_imbalanced_dataset(dataset_name)
    dataset_info = get_dataset_info(y)
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    return X, y, dataset_info, X_train, X_test, y_train, y_test, scaler

@st.cache_data
def load_pretrained_model(dataset_name, model_type_key, technique):
    """Load pre-trained model."""
    return model_loader.load_model(dataset_name, model_type_key, technique)

@st.cache_data
def load_pretrained_scaler(dataset_name):
    """Load pre-trained scaler."""
    return model_loader.load_scaler(dataset_name)

@st.cache_data(hash_funcs={pd.DataFrame: hash_dataframe, pd.Series: hash_series})
def apply_smote_to_data(X_train, y_train):
    """Apply SMOTE with caching."""
    smote_handler = SMOTEHandler(random_state=42)
    X_train_smote, y_train_smote = smote_handler.apply_smote(X_train, y_train)
    smote_info = smote_handler.get_class_distribution_info(y_train, y_train_smote)
    return X_train_smote, y_train_smote, smote_handler, smote_info

def predict_with_model(model, X_test):
    """Make predictions with a pre-trained model."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    return y_pred, y_pred_proba

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'introduction'

# Sidebar Navigation
st.sidebar.markdown("# 📚 SMOTE Virtual Lab")
st.sidebar.markdown("---")

# Navigation buttons in sidebar - stacked vertically
if st.sidebar.button("📖 Introduction", use_container_width=True, key="intro_btn"):
    st.session_state.current_page = 'introduction'
    st.rerun()

if st.sidebar.button("🎯 Objective", use_container_width=True, key="goals_btn"):
    st.session_state.current_page = 'objectives'
    st.rerun()

if st.sidebar.button("🔬 Simulation", use_container_width=True, key="sim_btn"):
    st.session_state.current_page = 'simulation'
    st.rerun()

st.sidebar.markdown("---")

# Initialize run_button to False (will be overridden on simulation page)
run_button = False

# ========== PAGE CONTENT ==========

# ========== INTRODUCTION PAGE ==========
if st.session_state.current_page == 'introduction':
    st.title("📖 Understanding SMOTE")
    
    st.markdown("""
    ### What is Class Imbalance?
    
    In many real-world datasets, classes are not equally distributed. For example:
    - Credit card fraud: ~0.1% fraudulent transactions
    - Disease detection: ~5% diseased patients
    - Network intrusion: ~1% intrusions
    
    This creates a **class imbalance problem** where standard ML algorithms are biased 
    toward the majority class.
    """)
    
    st.markdown("""
    ### SMOTE: Synthetic Minority Over-sampling Technique
    
    SMOTE is an algorithm that creates synthetic samples of the minority class by 
    interpolating between existing minority class samples.
    
    **Mathematical Formulation:**
    
    For a minority class sample $\\mathbf{x}_i$, SMOTE finds its $k$ nearest neighbors 
    and creates synthetic samples using:
    
    $$\\mathbf{x}_{synthetic} = \\mathbf{x}_i + \\lambda \\cdot (\\mathbf{x}_{neighbor} - \\mathbf{x}_i)$$
    
    where:
    - $\\mathbf{x}_i$ = a randomly selected minority class sample
    - $\\mathbf{x}_{neighbor}$ = one of its $k$ nearest minority class neighbors
    - $\\lambda$ = random value between 0 and 1
    
    **Result:** The minority class is balanced to match the majority class.
    """)
    
    st.markdown("""
    ### Key Equations
    
    **Class Imbalance Ratio:**
    $$IR = \\frac{n_{majority}}{n_{minority}}$$
    
    **Sampling Strategy:**
    $$N_{synthetic} = n_{minority} \\times (IR - 1)$$
    
    This generates exactly enough synthetic samples to balance the classes.
    """)
    
    st.markdown("""
    ### Comparison with Alternatives
    
    | Technique | Speed | Quality | Interpretability |
    |-----------|-------|---------|-----------------|
    | **SMOTE** | ⚡ Fast | ✓ Good | ✓ Clear |
    | **Random Over-sampling** | ⚡⚡ Very Fast | ✗ Poor | ✓ Clear |
    | **Random Under-sampling** | ⚡⚡ Very Fast | ✗ Poor | ✓ Clear |
    | **GAN** | 🐢 Slow | ✓✓ Excellent | ✗ Black Box |
    | **ADASYN** | ⚡ Fast | ✓ Good | ⚠ Medium |
    """)

# ========== OBJECTIVES PAGE ==========
elif st.session_state.current_page == 'objectives':
    st.title("🎯 Lab Objectives")
    
    st.markdown("""
    ### Learning Goals
    
    This virtual lab is designed to help you:
    
    **1. Understand Class Imbalance**
    - 🎓 Learn why imbalanced datasets are problematic
    - 📊 Visualize the impact on model performance
    - 🔍 Identify imbalance in real-world scenarios
    
    **2. Master the SMOTE Algorithm**
    - 🧠 Understand how synthetic samples are created
    - 📐 Learn the mathematical principles
    - 🔧 Know when and how to apply SMOTE
    
    **3. Evaluate Model Performance**
    - 📈 Use appropriate metrics (Recall, Precision, F1-Score)
    - 🎯 Understand accuracy limitations for imbalanced data
    - 📊 Interpret confusion matrices
    
    **4. Compare Techniques**
    - ⚖️ SMOTE vs. GAN approaches
    - ⏱️ Speed vs. Quality trade-offs
    - 💡 Make informed choices for your projects
    
    ### Expected Outcomes
    
    After completing this lab, you should be able to:
    
    ✅ Identify class imbalance in datasets\n
    ✅ Apply SMOTE to balance training data\n
    ✅ Evaluate results using appropriate metrics\n
    ✅ Compare different balancing techniques\n
    ✅ Make recommendations for handling imbalanced data\n
    """)

# ========== SIMULATION PAGE ==========
elif st.session_state.current_page == 'simulation':
    st.title("🔬 Interactive Analysis Suite")
    
    st.markdown("""
    <style>
    .header-subtitle {
        font-size: 1.1em;
        color: #666;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .config-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .config-header {
        font-size: 1.3em;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .dataset-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #e7f3ff;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #0366d6;
        height: 250px;
        box-sizing: border-box;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .model-comparison {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .model-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #e1e4e8;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    .model-title {
        font-size: 1.1em;
        font-weight: 600;
        color: #0366d6;
        margin-bottom: 0.5rem;
    }
    .model-feature {
        margin: 0.5rem 0;
        font-size: 0.95em;
        line-height: 1.4;
    }
    .feature-good {
        color: #28a745;
    }
    .feature-caution {
        color: #fd7e14;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        Explore how different machine learning techniques handle class imbalance with our interactive analysis suite.
        Select your dataset and model configuration below to begin the analysis.
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("")
    
    # ========== DATASET CONFIGURATION ==========
    with st.container():
        st.markdown("""
        <div class="config-section">
            <div class="config-header">📊 Step 1: Select Your Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get available datasets
        available_datasets = model_loader.get_available_datasets()
        
        if not available_datasets:
            st.error("❌ No pre-trained models found. Please run train_all_models.py first.")
            st.stop()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_dataset = st.selectbox(
                "Choose a Dataset",
                available_datasets,
                help="Select from four real-world imbalanced classification datasets",
                key="dataset_select"
            )
        
        # Dataset information and details
        dataset_descriptions = {
            "Attrition": {
                "domain": "Human Resources",
                "samples": "1,470",
                "features": "26",
                "imbalance": "5.20:1",
                "minority_pct": "16.12%",
                "description": "Employee attrition prediction dataset with HR metrics",
                "applications": "Workforce retention, employee segmentation"
            },
            "Bank": {
                "domain": "Finance",
                "samples": "45,211",
                "features": "7",
                "imbalance": "7.55:1",
                "minority_pct": "11.70%",
                "description": "Bank marketing campaign response dataset",
                "applications": "Customer targeting, campaign optimization"
            },
            "Credit Card": {
                "domain": "Fraud Detection",
                "samples": "284,807",
                "features": "30",
                "imbalance": "577.88:1",
                "minority_pct": "0.17%",
                "description": "Credit card fraud detection with extreme class imbalance",
                "applications": "Fraud prevention, anomaly detection"
            },
            "Diabetes": {
                "domain": "Healthcare",
                "samples": "768",
                "features": "8",
                "imbalance": "1.87:1",
                "minority_pct": "34.90%",
                "description": "Diabetes prediction dataset with minimal imbalance",
                "applications": "Disease prediction, patient screening"
            }
        }
        
        if selected_dataset in dataset_descriptions:
            desc = dataset_descriptions[selected_dataset]
            
            # Display dataset card with all information
            st.markdown(f"""
            <div class="dataset-card" style="background: #f0f2f6; border-left: 5px solid #667eea; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.2rem;">
                    <div>
                        <span style="font-size: 0.9em; color: #666; text-transform: uppercase; font-weight: 600;">Domain</span>
                        <div style="font-size: 1.4em; font-weight: 700; color: #0f1419;">{desc["domain"]}</div>
                    </div>
                    <div>
                        <span style="font-size: 0.9em; color: #666; text-transform: uppercase; font-weight: 600;">Samples</span>
                        <div style="font-size: 1.4em; font-weight: 700; color: #0f1419;">{desc["samples"]}</div>
                    </div>
                    <div>
                        <span style="font-size: 0.9em; color: #666; text-transform: uppercase; font-weight: 600;">Imbalance Ratio</span>
                        <div style="font-size: 1.4em; font-weight: 700; color: #0f1419;">{desc["imbalance"]}</div>
                    </div>
                    <div>
                        <span style="font-size: 0.9em; color: #666; text-transform: uppercase; font-weight: 600;">Minority %</span>
                        <div style="font-size: 1.4em; font-weight: 700; color: #0f1419;">{desc["minority_pct"]}</div>
                    </div>
                </div>
                <div style="border-top: 1px solid #ddd; padding-top: 1rem;">
                    <div style="margin-bottom: 0.8rem;">
                        <strong>📌 Description:</strong><br>
                        <span style="font-size: 0.95em; color: #555;">{desc["description"]}</span>
                    </div>
                    <div>
                        <strong>💡 Use Cases:</strong><br>
                        <span style="font-size: 0.95em; color: #555;">{desc["applications"]}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ========== MODEL CONFIGURATION ==========
    with st.container():
        st.markdown("""
        <div class="config-section">
            <div class="config-header">🤖 Step 2: Select Classification Algorithm</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model_type = st.radio(
                "Choose Algorithm",
                ["Random Forest", "Logistic Regression"],
                help="Select the machine learning algorithm",
                key="model_select"
            )
        
        with col2:
            st.markdown("""
            <div class="model-comparison">
            """, unsafe_allow_html=True)
            
            if model_type == "Random Forest":
                st.markdown("""
                <div class="model-card">
                    <div class="model-title">🌲 Random Forest Classifier</div>
                    <div class="model-feature"><span class="feature-good">✓ Handles non-linear patterns</span></div>
                    <div class="model-feature"><span class="feature-good">✓ Robust to feature scaling</span></div>
                    <div class="model-feature"><span class="feature-good">✓ Feature importance analysis</span></div>
                    <div class="model-feature"><span class="feature-caution">⚠ Slower inference time</span></div>
                    <div class="model-feature"><span class="feature-caution">⚠ Prone to overfitting</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="model-card">
                    <div class="model-title">📊 Logistic Regression</div>
                    <div class="model-feature"><span class="feature-good">✓ Fast training and inference</span></div>
                    <div class="model-feature"><span class="feature-good">✓ Highly interpretable</span></div>
                    <div class="model-feature"><span class="feature-good">✓ Probability outputs</span></div>
                    <div class="model-feature"><span class="feature-caution">⚠ Assumes linear relationships</span></div>
                    <div class="model-feature"><span class="feature-caution">⚠ Requires scaling</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("")
    
    # ========== ANALYSIS OPTIONS ==========
    with st.container():
        st.markdown("""
        <div class="config-section">
            <div class="config-header">⚙️ Step 3: Configure Analysis Options</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Original Data Analysis**
            
            Evaluates model performance on unbalanced training data. 
            This serves as the baseline for comparison.
            """)
            compare_techniques = True
        
        with col2:
            st.markdown("""
            **SMOTE Balancing**
            
            Synthetic Minority Over-sampling Technique creates synthetic 
            minority class samples before training.
            """)
        
        with col3:
            st.markdown("""
            **Optional GAN Training**
            
            After initial analysis, you can optionally train 
            Generative models for comparison (1-3 min).
            """)
    
    st.markdown("")
    
    # ========== EXECUTION SECTION ==========
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <strong>⚡ Performance Profile</strong><br>
                <span style="font-size: 0.9em;">All models are pre-trained and cached for instant inference.</span><br>
                <span style="font-size: 0.85em; color: #0366d6;">Typical analysis completion: &lt;2 seconds</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <strong>📈 Metrics Included</strong><br>
                <span style="font-size: 0.9em;">
                Accuracy, Precision, Recall,<br>
                F1-Score, ROC-AUC, Confusion Matrix
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <strong>✅ Outputs</strong><br>
                <span style="font-size: 0.9em;">
                Interactive visualizations,<br>
                detailed metrics, insights
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    st.markdown("")
    
    # ========== RUN BUTTON ==========
    col_button = st.columns([1])[0]
    
    run_button = st.button(
        "▶️  Run Analysis",
        use_container_width=True,
        key="run_button",
        help="Load pre-trained models and execute analysis"
    )
    
    st.markdown("")

    model_type_map = {
        "Random Forest": "random_forest",
        "Logistic Regression": "logistic_regression"
    }
    
    # Handle the run button click
    if run_button:
        with st.spinner("🔄 Loading pre-trained models and executing analysis..."):
            try:
                # Load data
                X, y, dataset_info, X_train, X_test, y_train, y_test, scaler = load_and_prep_data(selected_dataset)
                
                # Load pre-trained models
                model_type_key = model_type_map[model_type]
                model_original = load_pretrained_model(selected_dataset, model_type_key, "original")
                model_smote = load_pretrained_model(selected_dataset, model_type_key, "smote")
                
                # Get predictions
                y_pred_original, y_pred_proba_original = predict_with_model(model_original, X_test)
                y_pred_smote, y_pred_proba_smote = predict_with_model(model_smote, X_test)
                
                # Evaluate models
                metrics_original = ModelEvaluator.evaluate(y_test, y_pred_original, y_pred_proba_original)
                metrics_smote = ModelEvaluator.evaluate(y_test, y_pred_smote, y_pred_proba_smote)
                
                # Compute dataframes for stable display (prevents table flickering)
                feature_stats = X.iloc[:, :5].describe().T
                metrics_df_original = ModelEvaluator.get_metrics_dataframe(metrics_original)
                metrics_df_smote_computed = ModelEvaluator.get_metrics_dataframe(metrics_smote)
                comparison_df_computed = ModelEvaluator.compare_metrics(metrics_original, metrics_smote, "SMOTE")
                
                # Apply SMOTE for distribution info
                X_train_smote_temp, y_train_smote_temp, smote_handler_temp, smote_info_temp = apply_smote_to_data(X_train, y_train)
                dist_df_computed = SMOTEHandler.get_distribution_dataframe(y_train, y_train_smote_temp)
                details_df_computed = pd.DataFrame({
                    "Metric": ["Synthetic Samples Created", "Original Imbalance Ratio", "Post-SMOTE Ratio"],
                    "Value": [
                        str(smote_info_temp['Samples Added']),
                        smote_info_temp['Original Ratio'],
                        smote_info_temp['SMOTE Ratio']
                    ]
                })
                
                # Store in session state
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.dataset_info = dataset_info
                st.session_state.model_type = model_type
                st.session_state.model_type_key = model_type_key
                st.session_state.selected_dataset = selected_dataset
                st.session_state.model_original = model_original
                st.session_state.model_smote = model_smote
                st.session_state.y_pred_original = y_pred_original
                st.session_state.y_pred_proba_original = y_pred_proba_original
                st.session_state.y_pred_smote = y_pred_smote
                st.session_state.y_pred_proba_smote = y_pred_proba_smote
                st.session_state.metrics_original = metrics_original
                st.session_state.metrics_smote = metrics_smote
                st.session_state.compare_techniques = compare_techniques
                
                # Store computed dataframes to prevent flickering
                st.session_state.feature_stats_df = feature_stats
                st.session_state.metrics_df_original = metrics_df_original
                st.session_state.metrics_df_smote = metrics_df_smote_computed
                st.session_state.comparison_df = comparison_df_computed
                st.session_state.dist_df = dist_df_computed
                st.session_state.details_df = details_df_computed
                st.session_state.smote_info = smote_info_temp
                st.session_state.analysis_ready = True
                
            except FileNotFoundError as e:
                st.error(f"❌ Error loading models: {str(e)}\nPlease ensure train_all_models.py has been run.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# Main analysis section
if "analysis_ready" in st.session_state and st.session_state.analysis_ready:
    st.markdown("""
    <style>
    .section-divider {
        border-top: 3px solid #667eea;
        margin: 3rem 0 2rem 0;
    }
    .section-title {
        font-size: 1.8em;
        color: #0f1419;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .insight-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .warning-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ======== SECTION 1: EXPLORATORY DATA ANALYSIS ========
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('### 📊 Section 1: Dataset Overview & Class Imbalance Analysis')
    
    st.markdown("""
    This section provides a comprehensive analysis of the dataset's characteristics, 
    with particular focus on class distribution and imbalance metrics that are critical 
    for understanding model behavior and selection strategy.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Dataset Characteristics")
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.8rem 1.2rem;
            border-radius: 10px;
            margin-bottom: 0.6rem;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            min-height: 70px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-label {
            font-size: 0.75em;
            opacity: 0.9;
            font-weight: 600;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 1.6em;
            font-weight: 700;
            line-height: 1.1;
        }
        </style>
        """, unsafe_allow_html=True)
        
        info_df = pd.DataFrame(list(st.session_state.dataset_info.items()), 
                              columns=["Metric", "Value"])
        info_df["Value"] = info_df["Value"].astype(str)
        
        # Display each metric in a professional card box
        for idx, (_, row) in enumerate(info_df.iterrows()):
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{row["Metric"]}</div>
                <div class="metric-value">{row["Value"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add dataset preview option
        st.markdown("---")
        if st.checkbox(f"📋 Preview first 5 rows of {st.session_state.selected_dataset}", key="preview_dataset"):
            st.markdown("##### Dataset Preview")
            preview_df = st.session_state.X.head(5)
            st.dataframe(preview_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Class Distribution Analysis")
        class_counts = st.session_state.y.value_counts().sort_index()
        
        # Create visualization with pie chart below the histogram
        fig = plt.figure(figsize=(8, 10))
        gs = fig.add_gridspec(2, 1, hspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Bar plot (histogram)
        bars = ax1.bar(['Majority (0)', 'Minority (1)'], 
                      [class_counts[0], class_counts[1]], 
                      color=['#667eea', '#ff6b6b'], alpha=0.8, edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('Sample Count', fontsize=11, fontweight=600)
        ax1.set_title('Class Count Distribution', fontsize=12, fontweight=600)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontweight=600)
        
        # Pie chart (below histogram)
        colors = ['#667eea', '#ff6b6b']
        wedges, texts, autotexts = ax2.pie([class_counts[0], class_counts[1]], 
                                  labels=['Majority', 'Minority'],
                                  autopct='%1.1f%%',
                                  colors=colors,
                                  startangle=90,
                                  textprops={'fontsize': 11, 'fontweight': 600})
        ax2.set_title('Class Proportion', fontsize=12, fontweight=600)
        
        st.pyplot(fig)
    
    # Imbalance analysis insight
    imbalance_info = st.session_state.dataset_info
    if isinstance(imbalance_info.get('Imbalance Ratio'), str):
        ratio_str = imbalance_info['Imbalance Ratio']
        minority_pct = float(str(imbalance_info.get('Minority Class %', '0')).replace('%', ''))
    else:
        ratio_str = f"{imbalance_info.get('Imbalance Ratio', 0)}:1"
        minority_pct = float(imbalance_info.get('Minority Class %', 0))
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>📌 Imbalance Severity Assessment:</strong><br>
    The dataset exhibits a class imbalance ratio of <code>{ratio_str}</code> with the minority class 
    representing only <code>{minority_pct:.2f}%</code> of the total samples. This level of imbalance 
    can lead to models that achieve high accuracy by simply predicting the majority class while 
    completely missing minority class instances. Balanced evaluation metrics (Recall, Precision, F1-Score) 
    are therefore critical.
    </div>
    """, unsafe_allow_html=True)
    
    # Data statistics
    st.markdown("#### Feature Statistics (Sample of First 5 Features)")
    if "feature_stats_df" in st.session_state:
        st.dataframe(st.session_state.feature_stats_df, use_container_width=True)
    else:
        feature_stats = st.session_state.X.iloc[:, :5].describe().T
        st.dataframe(feature_stats, use_container_width=True)
    
    # ======== SECTION 2: MODEL PERFORMANCE (ORIGINAL DATA) ========
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('### 🤖 Section 2: Baseline Model Performance (Original Imbalanced Data)')
    
    st.markdown("""
    This section evaluates the pre-trained model performance on the original, imbalanced dataset.
    These metrics serve as the baseline for comparing with SMOTE-balanced approach.
    **Note:** The original imbalanced data often leads to high accuracy but poor minority class recall.
    """)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_original = st.session_state.metrics_original
    
    with col1:
        st.metric(
            "Accuracy",
            f"{metrics_original['Accuracy']:.4f}",
            help="Overall correctness: (TP+TN)/(TP+TN+FP+FN)"
        )
    with col2:
        st.metric(
            "Precision",
            f"{metrics_original['Precision']:.4f}",
            help="Of predicted positives, how many are actually positive: TP/(TP+FP)"
        )
    with col3:
        st.metric(
            "Recall",
            f"{metrics_original['Recall']:.4f}",
            help="Of actual positives, how many did we catch: TP/(TP+FN) [Critical for imbalanced data]"
        )
    with col4:
        st.metric(
            "F1-Score",
            f"{metrics_original['F1-Score']:.4f}",
            help="Harmonic mean of Precision and Recall: 2·(P·R)/(P+R)"
        )
    with col5:
        st.metric(
            "ROC-AUC",
            f"{metrics_original.get('ROC-AUC', 0.0):.4f}",
            help="Area under ROC curve: probabilistic ranking ability"
        )
    
    # Analysis details
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Confusion Matrix")
        cm_original = st.session_state.metrics_original['Confusion Matrix']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative (0)', 'Positive (1)'],
                   yticklabels=['Negative (0)', 'Positive (1)'],
                   cbar_kws={'label': 'Count'},
                   ax=ax, linewidths=1, linecolor='gray')
        ax.set_ylabel('True Label', fontweight=600)
        ax.set_xlabel('Predicted Label', fontweight=600)
        ax.set_title('Original Model Confusion Matrix', fontweight=600, fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        cm_original_array = np.array(cm_original)
        tn, fp, fn, tp = cm_original_array.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        st.markdown(f"""
        **Confusion Matrix Breakdown:**
        - **True Negatives (TN):** {tn} - Correctly identified majority examples
        - **False Positives (FP):** {fp} - Majority misclassified as minority  
        - **False Negatives (FN):** {fn} - Minority missed (problematic!)
        - **True Positives (TP):** {tp} - Correctly identified minority examples
        
        **Sensitivity (Recall):** {sensitivity:.4f} | **Specificity:** {specificity:.4f}
        """)
    
    with col2:
        st.markdown("#### Performance Metrics Table")
        if "metrics_df_original" in st.session_state:
            st.dataframe(st.session_state.metrics_df_original, use_container_width=True, hide_index=True)
        else:
            metrics_df = ModelEvaluator.get_metrics_dataframe(st.session_state.metrics_original)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Add insight
        if metrics_original['Recall'] < 0.6:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Critical Finding:</strong> Low recall indicates the model is missing many 
            minority class instances. This is typical on imbalanced data and often unacceptable 
            for real-world applications (fraud detection, disease diagnosis, etc.).
            </div>
            """, unsafe_allow_html=True)
    
    # ======== SECTION 3: APPLY SMOTE ========
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('### ⚖️ Section 3: Applying SMOTE for Class Balancing')
    
    st.markdown("""
    **SMOTE (Synthetic Minority Over-sampling Technique)** addresses class imbalance by creating 
    synthetic samples of the minority class. For each minority instance, SMOTE:
    1. Finds k nearest neighbors in the minority class (k=5 by default)
    2. Randomly selects one neighbor
    3. Creates a new synthetic sample along the line connecting the two points
    
    **Mathematical Formula:**
    """)
    
    st.latex(r"x_{\text{synthetic}} = x_i + \lambda(x_{\text{neighbor}} - x_i), \quad \lambda \in [0, 1]")
    
    st.markdown("""
    This approach preserves feature relationships while increasing minority class representation.
    """)
    
    with st.spinner("🔄 Applying SMOTE to training data..."):
        X_train_smote, y_train_smote, smote_handler, smote_info = apply_smote_to_data(
            st.session_state.X_train, 
            st.session_state.y_train
        )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Class Distribution Before & After SMOTE")
        if "dist_df" in st.session_state:
            st.dataframe(st.session_state.dist_df, use_container_width=True, hide_index=True)
        else:
            dist_df = SMOTEHandler.get_distribution_dataframe(
                st.session_state.y_train, 
                y_train_smote
            )
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### SMOTE Application Summary")
        if "details_df" in st.session_state:
            st.dataframe(st.session_state.details_df, use_container_width=True, hide_index=True)
        else:
            details = pd.DataFrame({
                "Metric": ["Synthetic Samples Created", "Original Imbalance Ratio", "Post-SMOTE Ratio"],
                "Value": [
                    str(smote_info['Samples Added']),
                    smote_info['Original Ratio'],
                    smote_info['SMOTE Ratio']
                ]
            })
            st.dataframe(details, use_container_width=True, hide_index=True)
    
    # Visualization - Class distribution comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sample Count Evolution")
        fig, ax = plt.subplots(figsize=(7, 5))
        classes = ['Majority (0)', 'Minority (1)']
        original = [
            smote_info['Original Distribution'][0],
            smote_info['Original Distribution'][1]
        ]
        after_smote = [
            smote_info['SMOTE Distribution'][0],
            smote_info['SMOTE Distribution'][1]
        ]
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original, width, label='Before SMOTE', 
                      color='#667eea', alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, after_smote, width, label='After SMOTE',
                      color='#51cf66', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Sample Count', fontweight=600, fontsize=11)
        ax.set_title('SMOTE Impact on Class Distribution', fontweight=600, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight=600)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### Imbalance Ratio Reduction")
        fig, ax = plt.subplots(figsize=(7, 5))
        
        before_ratio_str = smote_info['Original Ratio']
        after_ratio_str = smote_info['SMOTE Ratio']
        
        before_ratio_val = float(before_ratio_str.split(':')[0])
        after_ratio_val = float(after_ratio_str.split(':')[0])
        
        ratios = [before_ratio_val, after_ratio_val]
        labels = [f'Before\n({before_ratio_str})', f'After\n({after_ratio_str})']
        colors = ['#ff6b6b', '#51cf66']
        
        bars = ax.bar(labels, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2, width=0.6)
        
        ax.set_ylabel('Imbalance Ratio (Majority/Minority)', fontweight=600, fontsize=11)
        ax.set_title('Imbalance Ratio Reduction', fontweight=600, fontsize=12)
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Balance (1:1)')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.2f}:1', ha='center', va='bottom', fontsize=11, fontweight=600)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Success insight
    reduction_pct = ((before_ratio_val - after_ratio_val) / before_ratio_val * 100) if before_ratio_val > 0 else 0
    st.markdown(f"""
    <div class="success-box">
    <strong>✅ SMOTE Successfully Applied:</strong><br>
    Created {smote_info['Samples Added']:,} synthetic minority samples, reducing imbalance ratio from 
    <code>{before_ratio_str}</code> to <code>{after_ratio_str}</code> (<strong>{reduction_pct:.1f}% reduction</strong>).
    Training data is now better balanced, allowing models to learn minority class patterns.
    </div>
    """, unsafe_allow_html=True)
    
    # ======== SECTION 4: PRE-TRAINED MODEL PERFORMANCE (SMOTE-BALANCED DATA) ========
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('### 🤖 Section 4: Improved Model Performance (SMOTE-Balanced Data)')
    
    st.markdown("""
    After applying SMOTE, we retrain the model on the balanced training data and evaluate 
    performance on the same test set. This demonstrates the impact of balanced training on 
    both majority and minority class detection rates.
    """)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_smote = st.session_state.metrics_smote
    
    with col1:
        st.metric(
            "Accuracy",
            f"{metrics_smote['Accuracy']:.4f}",
            delta=f"{(metrics_smote['Accuracy'] - st.session_state.metrics_original['Accuracy']):.4f}"
        )
    with col2:
        st.metric(
            "Precision",
            f"{metrics_smote['Precision']:.4f}",
            delta=f"{(metrics_smote['Precision'] - st.session_state.metrics_original['Precision']):.4f}"
        )
    with col3:
        st.metric(
            "Recall",
            f"{metrics_smote['Recall']:.4f}",
            delta=f"{(metrics_smote['Recall'] - st.session_state.metrics_original['Recall']):.4f}"
        )
    with col4:
        st.metric(
            "F1-Score",
            f"{metrics_smote['F1-Score']:.4f}",
            delta=f"{(metrics_smote['F1-Score'] - st.session_state.metrics_original['F1-Score']):.4f}"
        )
    with col5:
        st.metric(
            "ROC-AUC",
            f"{metrics_smote.get('ROC-AUC', 0.0):.4f}",
            delta=f"{(metrics_smote.get('ROC-AUC', 0.0) - st.session_state.metrics_original.get('ROC-AUC', 0.0)):.4f}"
        )
    
    # Analysis details
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Confusion Matrix")
        cm_smote = st.session_state.metrics_smote['Confusion Matrix']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['Negative (0)', 'Positive (1)'],
                   yticklabels=['Negative (0)', 'Positive (1)'],
                   cbar_kws={'label': 'Count'},
                   ax=ax, linewidths=1, linecolor='gray')
        ax.set_ylabel('True Label', fontweight=600)
        ax.set_xlabel('Predicted Label', fontweight=600)
        ax.set_title('SMOTE-Balanced Model Confusion Matrix', fontweight=600, fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        cm_smote_array = np.array(cm_smote)
        tn, fp, fn, tp = cm_smote_array.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        st.markdown(f"""
        **Confusion Matrix Breakdown:**
        - **True Negatives (TN):** {tn} - Correctly identified majority examples
        - **False Positives (FP):** {fp} - Majority misclassified as minority
        - **False Negatives (FN):** {fn} - Minority missed (significantly reduced!)
        - **True Positives (TP):** {tp} - Correctly identified minority examples
        
        **Sensitivity (Recall):** {sensitivity:.4f} | **Specificity:** {specificity:.4f}
        """)
    
    with col2:
        st.markdown("#### Performance Metrics Table")
        if "metrics_df_smote" in st.session_state:
            st.dataframe(st.session_state.metrics_df_smote, use_container_width=True, hide_index=True)
        else:
            metrics_df_smote = ModelEvaluator.get_metrics_dataframe(st.session_state.metrics_smote)
            st.dataframe(metrics_df_smote, use_container_width=True, hide_index=True)
        
        # Add insight
        if metrics_smote['Recall'] > st.session_state.metrics_original['Recall']:
            st.markdown("""
            <div class="success-box">
            <strong>✅ Significant Improvement:</strong> SMOTE training substantially improved 
            minority class recall. The model now catches more minority instances that would have 
            been missed on imbalanced data.
            </div>
            """, unsafe_allow_html=True)
    
    # ======== SECTION 5: PERFORMANCE COMPARISON ========
    if st.session_state.compare_techniques:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('### 📈 Section 5: Comprehensive Performance Comparison')
        
        st.markdown("""
        This section provides a detailed side-by-side comparison of model performance between 
        the original imbalanced approach and the SMOTE-balanced approach across all key metrics.
        """)
        
        st.markdown("#### Detailed Metrics Comparison")
        if "comparison_df" in st.session_state:
            st.dataframe(st.session_state.comparison_df, use_container_width=True, hide_index=True)
        else:
            comparison_df = ModelEvaluator.compare_metrics(
                st.session_state.metrics_original, 
                st.session_state.metrics_smote, 
                "SMOTE"
            )
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visualization - Grouped comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Metrics Comparison Chart")
            fig, ax = plt.subplots(figsize=(9, 6))
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            original_vals = [
                st.session_state.metrics_original['Accuracy'],
                st.session_state.metrics_original['Precision'],
                st.session_state.metrics_original['Recall'],
                st.session_state.metrics_original['F1-Score'],
                st.session_state.metrics_original.get('ROC-AUC', 0.0)
            ]
            smote_vals = [
                st.session_state.metrics_smote['Accuracy'],
                st.session_state.metrics_smote['Precision'],
                st.session_state.metrics_smote['Recall'],
                st.session_state.metrics_smote['F1-Score'],
                st.session_state.metrics_smote.get('ROC-AUC', 0.0)
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, original_vals, width, label='Original (Imbalanced)',
                          color='#667eea', alpha=0.8, edgecolor='black', linewidth=1.2)
            bars2 = ax.bar(x + width/2, smote_vals, width, label='SMOTE (Balanced)',
                          color='#51cf66', alpha=0.8, edgecolor='black', linewidth=1.2)
            
            ax.set_ylabel('Score', fontweight=600, fontsize=11)
            ax.set_title('Original vs SMOTE: All Metrics', fontweight=600, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim([0, 1.15])
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight=600)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Performance Improvement Visualization")
            fig, ax = plt.subplots(figsize=(9, 6))
            
            improvements = [
                (st.session_state.metrics_smote['Accuracy'] - st.session_state.metrics_original['Accuracy']) * 100,
                (st.session_state.metrics_smote['Precision'] - st.session_state.metrics_original['Precision']) * 100,
                (st.session_state.metrics_smote['Recall'] - st.session_state.metrics_original['Recall']) * 100,
                (st.session_state.metrics_smote['F1-Score'] - st.session_state.metrics_original['F1-Score']) * 100,
                (st.session_state.metrics_smote.get('ROC-AUC', 0.0) - st.session_state.metrics_original.get('ROC-AUC', 0.0)) * 100
            ]
            
            colors = ['#51cf66' if x >= 0 else '#ff6b6b' for x in improvements]
            bars = ax.barh(metrics_names, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            ax.set_xlabel('Improvement (%)', fontweight=600, fontsize=11)
            ax.set_title('SMOTE Performance Improvement', fontweight=600, fontsize=12)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar, val in zip(bars, improvements):
                x_pos = val + (1 if val >= 0 else -1)
                ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
                       f'{val:+.2f}%', ha='left' if val >= 0 else 'right',
                       va='center', fontsize=10, fontweight=600)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Summary insights
        avg_improvement = np.mean([improvements[i] for i in range(len(improvements)) if i != 0])  # Exclude accuracy
        
        if avg_improvement > 0:
            insight_text = f"SMOTE training significantly improved recall-oriented metrics by an average of {avg_improvement:.1f}%, " \
                          f"making the model much more reliable for minority class detection."
            col_insight = st.columns([1])[0]
            st.markdown(f"""
            <div class="success-box">
            <strong>✅ Comprehensive Improvement:</strong><br>
            {insight_text} This is particularly valuable for applications where missing minority instances 
            is costly (fraud detection, disease diagnosis, anomaly detection).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
            <strong>📊 Comparative Analysis:</strong><br>
            The models show different trade-offs. Original model prioritizes majority accuracy 
            while SMOTE model improves minority class detection (Recall). The choice depends on 
            your application's specific requirements.
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(fig)
        
        # Key Insights
        st.subheader("📝 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recall_improvement = (st.session_state.metrics_smote['Recall'] - st.session_state.metrics_original['Recall']) * 100
            st.info(f"**Recall Improvement:** {recall_improvement:+.2f}%\n\n"
                   f"SMOTE significantly improves minority class detection!")
        
        with col2:
            f1_improvement = (st.session_state.metrics_smote['F1-Score'] - st.session_state.metrics_original['F1-Score']) * 100
            st.info(f"**F1-Score Improvement:** {f1_improvement:+.2f}%\n\n"
                   f"Better balance between precision and recall!")
        
        with col3:
            st.info(f"**SMOTE Advantage:**\n\nPre-trained models allow instant comparison without training delays!")
    
    # ======== SECTION 6: GAN COMPARISON ========
    st.header("🎨 Section 6: Advanced GAN Comparison")
    
    st.markdown("""
    GAN (Generative Adversarial Network) models provide another approach to handling class imbalance.
    If you want to see how GAN compares to SMOTE, you can train GAN models on-demand.
    
    **Note:** GAN training takes 1-3 minutes per dataset depending on size.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **SMOTE vs GAN:**
        - **SMOTE:** Fast, deterministic, stable
        - **GAN:** Complex, flexible, may generate higher-quality samples
        """)
    
    with col2:
        train_gan = st.checkbox(
            "🎨 Train GAN Models for Comparison",
            value=False,
            help="Enable GAN training (adds 1-3 minutes)"
        )
    
    if train_gan:
        st.warning("⏳ GAN training is in progress. This may take several minutes...")
        
        with st.spinner("Training GAN models..."):
            try:
                from utils.gan_handler import GANHandler
                import tensorflow as tf
                
                # Debug: Check if TensorFlow is available
                st.session_state.tf_version = tf.__version__
                
                # Train GAN and get results
                gan_handler = GANHandler(epochs=30, random_state=42)
                X_train_gan, y_train_gan, _ = gan_handler.apply_gan(
                    st.session_state.X_train, 
                    st.session_state.y_train, 
                    verbose=False
                )
                
                # Train model on GAN data
                from utils.models import ClassificationModel
                model_gan = ClassificationModel(
                    model_type=st.session_state.model_type_key,
                    random_state=42
                )
                model_gan.train(X_train_gan, y_train_gan)
                
                # Get predictions
                y_pred_gan, y_pred_proba_gan = predict_with_model(model_gan, st.session_state.X_test)
                metrics_gan = ModelEvaluator.evaluate(st.session_state.y_test, y_pred_gan, y_pred_proba_gan)
                
                st.session_state.metrics_gan = metrics_gan
                st.session_state.model_gan = model_gan
                st.session_state.y_pred_gan = y_pred_gan
                st.session_state.gan_trained = True
                
                st.success("✅ GAN training complete!")
                
                # Show debug info
                if "tf_version" in st.session_state:
                    st.info(f"🔧 TensorFlow version: {st.session_state.tf_version}")
            except ImportError as e:
                st.error(f"❌ GAN Import Error: TensorFlow may not be properly installed.\n\nDetails: {str(e)}\n\nPlease ensure tensorflow==2.13.0 is installed.")
            except Exception as e:
                import traceback
                st.error(f"❌ GAN training failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
        
        if st.session_state.get('gan_trained', False):
            st.subheader("🎨 GAN Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{st.session_state.metrics_gan['Accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{st.session_state.metrics_gan['Precision']:.4f}")
            with col3:
                st.metric("Recall", f"{st.session_state.metrics_gan['Recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{st.session_state.metrics_gan['F1-Score']:.4f}")
            
            # Three-way comparison
            st.subheader("📊 Three-Way Performance Comparison")
            
            comparison_three_way = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Original": [
                    f"{st.session_state.metrics_original['Accuracy']:.4f}",
                    f"{st.session_state.metrics_original['Precision']:.4f}",
                    f"{st.session_state.metrics_original['Recall']:.4f}",
                    f"{st.session_state.metrics_original['F1-Score']:.4f}"
                ],
                "SMOTE": [
                    f"{st.session_state.metrics_smote['Accuracy']:.4f}",
                    f"{st.session_state.metrics_smote['Precision']:.4f}",
                    f"{st.session_state.metrics_smote['Recall']:.4f}",
                    f"{st.session_state.metrics_smote['F1-Score']:.4f}"
                ],
                "GAN": [
                    f"{st.session_state.metrics_gan['Accuracy']:.4f}",
                    f"{st.session_state.metrics_gan['Precision']:.4f}",
                    f"{st.session_state.metrics_gan['Recall']:.4f}",
                    f"{st.session_state.metrics_gan['F1-Score']:.4f}"
                ]
            })
            
            st.dataframe(comparison_three_way, use_container_width=True, hide_index=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                original_vals = [
                    st.session_state.metrics_original['Accuracy'],
                    st.session_state.metrics_original['Precision'],
                    st.session_state.metrics_original['Recall'],
                    st.session_state.metrics_original['F1-Score']
                ]
                smote_vals = [
                    st.session_state.metrics_smote['Accuracy'],
                    st.session_state.metrics_smote['Precision'],
                    st.session_state.metrics_smote['Recall'],
                    st.session_state.metrics_smote['F1-Score']
                ]
                gan_vals = [
                    st.session_state.metrics_gan['Accuracy'],
                    st.session_state.metrics_gan['Precision'],
                    st.session_state.metrics_gan['Recall'],
                    st.session_state.metrics_gan['F1-Score']
                ]
                
                x = np.arange(len(metrics_names))
                width = 0.25
                
                ax.bar(x - width, original_vals, width, label='Original', color='#1f77b4')
                ax.bar(x, smote_vals, width, label='SMOTE', color='#2ca02c')
                ax.bar(x + width, gan_vals, width, label='GAN', color='#d62728')
                
                ax.set_ylabel('Score')
                ax.set_title('Three-Way Comparison: Original vs SMOTE vs GAN')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim([0, 1.1])
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                smote_improvements = [
                    (st.session_state.metrics_smote['Accuracy'] - st.session_state.metrics_original['Accuracy']) * 100,
                    (st.session_state.metrics_smote['Precision'] - st.session_state.metrics_original['Precision']) * 100,
                    (st.session_state.metrics_smote['Recall'] - st.session_state.metrics_original['Recall']) * 100,
                    (st.session_state.metrics_smote['F1-Score'] - st.session_state.metrics_original['F1-Score']) * 100
                ]
                
                gan_improvements = [
                    (st.session_state.metrics_gan['Accuracy'] - st.session_state.metrics_original['Accuracy']) * 100,
                    (st.session_state.metrics_gan['Precision'] - st.session_state.metrics_original['Precision']) * 100,
                    (st.session_state.metrics_gan['Recall'] - st.session_state.metrics_original['Recall']) * 100,
                    (st.session_state.metrics_gan['F1-Score'] - st.session_state.metrics_original['F1-Score']) * 100
                ]
                
                x = np.arange(len(metrics_names))
                width = 0.35
                
                ax.bar(x - width/2, smote_improvements, width, label='SMOTE', color='#2ca02c')
                ax.bar(x + width/2, gan_improvements, width, label='GAN', color='#d62728')
                
                ax.set_ylabel('Improvement (%)')
                ax.set_title('Improvement Over Original Model')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics_names, rotation=45, ha='right')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Recommendation
            st.header("🎯 SMOTE vs GAN Recommendation")
            
            smote_f1 = st.session_state.metrics_smote['F1-Score']
            gan_f1 = st.session_state.metrics_gan['F1-Score']
            original_f1 = st.session_state.metrics_original['F1-Score']
            
            if smote_f1 > gan_f1:
                recommendation = "SMOTE"
                difference = (smote_f1 - gan_f1) * 100
            else:
                recommendation = "GAN"
                difference = (gan_f1 - smote_f1) * 100
            
            st.success(f"""
            ### 📌 Recommendation: **{recommendation}** performs better
            
            **F1-Score Comparison:**
            - Original: {original_f1:.4f}
            - SMOTE: {smote_f1:.4f}
            - GAN: {gan_f1:.4f}
            
            **Winner's Advantage:** {difference:.2f} percentage points
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SMOTE Strengths")
                st.markdown("""
                ✅ Fast training (seconds)\n
                ✅ Deterministic results\n
                ✅ Easy to interpret\n
                ✅ Low computational overhead
                """)
            
            with col2:
                st.subheader("GAN Strengths")
                st.markdown("""
                ✅ Complex pattern learning\n
                ✅ High-quality samples\n
                ✅ Flexible approach\n
                ✅ Modern deep learning
                """)
    
    # ======== SUMMARY ========
    st.header("✅ Analysis Summary")
    
    st.markdown(f"""
    ### Dataset: {st.session_state.selected_dataset}
    ### Model: {st.session_state.model_type}
    
    **Key Findings:**
    
    The analysis demonstrates how SMOTE balancing improves model performance on imbalanced datasets.
    Pre-trained models allow for instant comparison without computational overhead.
    
    **Next Steps:**
    - Try different datasets to see how SMOTE performs
    - Compare model types (Random Forest vs Logistic Regression)
    - Analyze which techniques work best for different imbalance ratios
    """)

else:
    st.info("👈 Configure your analysis settings above and click the 'Run Analysis' button to start!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>SMOTE Virtual Lab</strong> | Demonstrating Class Imbalance Handling Techniques</p>
    <p style='font-size: 0.9em; color: gray;'>Made with Streamlit | SMOTE vs GAN Comparison</p>
</div>
""", unsafe_allow_html=True)
