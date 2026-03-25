import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_imbalanced_dataset, get_dataset_info, prepare_data
from utils.models import ClassificationModel, ModelEvaluator
from utils.smote_handler import SMOTEHandler
from utils.gan_handler import GANHandler
import warnings

warnings.filterwarnings('ignore')

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

@st.cache_data(hash_funcs={pd.DataFrame: hash_dataframe, pd.Series: hash_series})
def train_original_model(model_type_key, X_train, y_train, X_test, y_test):
    """Train model on original data with caching."""
    model = ClassificationModel(model_type=model_type_key, random_state=42)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_pred_proba)
    return model, y_pred, y_pred_proba, metrics

@st.cache_data(hash_funcs={pd.DataFrame: hash_dataframe, pd.Series: hash_series})
def apply_smote_to_data(X_train, y_train):
    """Apply SMOTE with caching."""
    smote_handler = SMOTEHandler(random_state=42)
    X_train_smote, y_train_smote = smote_handler.apply_smote(X_train, y_train)
    smote_info = smote_handler.get_class_distribution_info(y_train, y_train_smote)
    return X_train_smote, y_train_smote, smote_handler, smote_info

@st.cache_data(hash_funcs={pd.DataFrame: hash_dataframe, pd.Series: hash_series})
def train_smote_model(model_type_key, X_train_smote, y_train_smote, X_test, y_test):
    """Train model on SMOTE-balanced data with caching."""
    model = ClassificationModel(model_type=model_type_key, random_state=42)
    model.train(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_pred_proba)
    return model, y_pred, y_pred_proba, metrics

@st.cache_data(hash_funcs={pd.DataFrame: hash_dataframe, pd.Series: hash_series})
def apply_gan_to_data(X_train, y_train, X_test, y_test, model_type_key):
    """Apply GAN and train model with caching."""
    gan_handler = GANHandler(epochs=50, random_state=42)
    X_train_gan, y_train_gan, _ = gan_handler.apply_gan(X_train, y_train, verbose=False)
    
    model = ClassificationModel(model_type=model_type_key, random_state=42)
    model.train(X_train_gan, y_train_gan)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_pred_proba)
    return model, y_pred, y_pred_proba, metrics

# Set page config
st.set_page_config(
    page_title="SMOTE Virtual Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.title("🔬 Simulation Configuration")
    
    st.markdown("Configure your simulation parameters below to run the class imbalance analysis.")
    st.markdown("---")
    
    # Create a professional container layout
    with st.container():
        # Dataset Configuration Section
        st.subheader("📊 Data Configuration")
        
        datasets = [
            "Credit Card Fraud",
            "Disease Detection",
            "Network Intrusion",
            "Rare Event Prediction"
        ]
        
        selected_dataset = st.selectbox(
            "Select Dataset",
            datasets,
            help="Choose an imbalanced dataset to analyze",
            key="dataset_select"
        )
        
        dataset_descriptions = {
            "Credit Card Fraud": "10,000 samples | 2% minority | Extreme imbalance scenario",
            "Disease Detection": "8,000 samples | 5% minority | Medical domain",
            "Network Intrusion": "12,000 samples | 4% minority | Cybersecurity domain",
            "Rare Event Prediction": "9,000 samples | 8% minority | Moderate imbalance"
        }
        st.caption(f"🔹 {dataset_descriptions[selected_dataset]}")
        
    st.markdown("")
    
    with st.container():
        # Model Configuration Section
        st.subheader("🤖 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.radio(
                "Select Classification Model",
                ["Random Forest", "Logistic Regression"],
                help="Choose the ML algorithm",
                key="model_select"
            )
        
        with col2:
            model_info = {
                "Random Forest": "✓ Better for complex patterns\n✓ Handles non-linearity\n✗ Slower training",
                "Logistic Regression": "✓ Faster training\n✓ More interpretable\n✗ Assumes linearity"
            }
            st.info(model_info[model_type])
    
    st.markdown("")
    
    with st.container():
        # Advanced Options Section
        st.subheader("⚙️ Advanced Options")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            compare_with_gan = st.checkbox(
                "🎨 Compare with GAN Balancing",
                value=False,
                help="Enable advanced GAN comparison (adds 1-3 minutes)",
                key="gan_option"
            )
        
        with col2:
            if compare_with_gan:
                st.warning("⏱️ Extended runtime")
            else:
                st.success("⏱️ Standard runtime")
    
    st.markdown("")
    
    with st.container():
        # Runtime Estimate
        st.info(
            "⏱️ **Estimated Runtime:**\n"
            "• Without GAN: ~30 seconds\n"
            "• With GAN: ~2-3 minutes"
        )
    
    st.markdown("")
    
    # Run Button
    col_button_left, col_button_right = st.columns([2, 1])
    
    with col_button_left:
        run_button = st.button("🚀 Run Simulation", use_container_width=True, key="run_button")
    
    st.markdown("")

    model_type_map = {
        "Random Forest": "random_forest",
        "Logistic Regression": "logistic_regression"
    }
    
    # Handle the run button click
    if run_button:
        with st.spinner("Loading and preparing data..."):
            # Use cached data loading
            X, y, dataset_info, X_train, X_test, y_train, y_test, scaler = load_and_prep_data(selected_dataset)
            
            # Store in session state
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.dataset_info = dataset_info
            st.session_state.model_type = model_type
            st.session_state.model_type_key = model_type_map[model_type]
            st.session_state.compare_with_gan = compare_with_gan
            st.session_state.selected_dataset = selected_dataset
            st.session_state.analysis_ready = True

# Main analysis section
if "analysis_ready" in st.session_state and st.session_state.analysis_ready:
    
    # ======== SECTION 1: EXPLORATORY DATA ANALYSIS ========
    st.header("📊 Section 1: Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        info_df = pd.DataFrame(list(st.session_state.dataset_info.items()), 
                              columns=["Metric", "Value"])
        # Convert all values to string to prevent Arrow serialization issues
        info_df["Value"] = info_df["Value"].astype(str)
        st.table(info_df)
    
    with col2:
        st.subheader("Class Distribution (Original)")
        class_counts = st.session_state.y.value_counts().sort_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Bar plot
        ax1.bar(['Majority (0)', 'Minority (1)'], 
               [class_counts[0], class_counts[1]], 
               color=['#1f77b4', '#ff7f0e'])
        ax1.set_ylabel('Sample Count')
        ax1.set_title('Class Distribution')
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        ax2.pie([class_counts[0], class_counts[1]], 
               labels=['Majority', 'Minority'],
               autopct='%1.1f%%',
               colors=['#1f77b4', '#ff7f0e'])
        ax2.set_title('Class Proportion')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Data statistics
    st.subheader("Feature Statistics (First 5 Features)")
    feature_stats = st.session_state.X.iloc[:, :5].describe().T
    st.dataframe(feature_stats, use_container_width=True)
    
    # ======== SECTION 2: MODEL TRAINING (ORIGINAL DATA) ========
    st.header("🤖 Section 2: Model Training on Original Data")
    
    with st.spinner("Training model on original dataset..."):
        model_original, y_pred_original, y_pred_proba_original, metrics_original = train_original_model(
            st.session_state.model_type_key,
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.X_test,
            st.session_state.y_test
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics_original['Accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics_original['Precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics_original['Recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics_original['F1-Score']:.4f}")
    
    # Confusion matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix (Original)")
        cm_original = metrics_original['Confusion Matrix']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar=True, ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Original Model Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Metrics Comparison Table")
        metrics_df = ModelEvaluator.get_metrics_dataframe(metrics_original)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # ======== SECTION 3: APPLY SMOTE ========
    st.header("⚖️ Section 3: Applying SMOTE to Balance the Dataset")
    
    with st.spinner("Applying SMOTE..."):
        X_train_smote, y_train_smote, smote_handler, smote_info = apply_smote_to_data(
            st.session_state.X_train, 
            st.session_state.y_train
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution Before & After SMOTE")
        dist_df = SMOTEHandler.get_distribution_dataframe(
            st.session_state.y_train, 
            y_train_smote
        )
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("SMOTE Details")
        details = pd.DataFrame({
            "Metric": ["Samples Added", "Original Ratio", "SMOTE Ratio"],
            "Value": [
                str(smote_info['Samples Added']),
                smote_info['Original Ratio'],
                smote_info['SMOTE Ratio']
            ]
        })
        st.dataframe(details, use_container_width=True, hide_index=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
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
        
        ax.bar(x - width/2, original, width, label='Before SMOTE', color='#1f77b4')
        ax.bar(x + width/2, after_smote, width, label='After SMOTE', color='#2ca02c')
        
        ax.set_ylabel('Sample Count')
        ax.set_title('Class Distribution: Before vs After SMOTE')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        before_ratio_val = float(smote_info['Original Ratio'].split(':')[0])
        after_ratio_val = float(smote_info['SMOTE Ratio'].split(':')[0])
        
        ax.bar(['Original Ratio', 'After SMOTE'], 
              [before_ratio_val, after_ratio_val],
              color=['#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Imbalance Ratio (Majority:Minority)')
        ax.set_title('Imbalance Ratio Reduction')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Perfect Balance')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    # ======== SECTION 4: MODEL TRAINING (SMOTE-BALANCED DATA) ========
    st.header("🤖 Section 4: Model Training on SMOTE-Balanced Data")
    
    with st.spinner("Training model on SMOTE-balanced dataset..."):
        model_smote, y_pred_smote, y_pred_proba_smote, metrics_smote = train_smote_model(
            st.session_state.model_type_key,
            X_train_smote,
            y_train_smote,
            st.session_state.X_test,
            st.session_state.y_test
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics_smote['Accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics_smote['Precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics_smote['Recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics_smote['F1-Score']:.4f}")
    
    # Confusion matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix (SMOTE-Balanced)")
        cm_smote = metrics_smote['Confusion Matrix']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar=True, ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('SMOTE-Balanced Model Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Metrics Comparison Table")
        metrics_df_smote = ModelEvaluator.get_metrics_dataframe(metrics_smote)
        st.dataframe(metrics_df_smote, use_container_width=True, hide_index=True)
    
    # ======== SECTION 5: PERFORMANCE COMPARISON ========
    st.header("📈 Section 5: Performance Comparison (Before vs After SMOTE)")
    
    comparison_df = ModelEvaluator.compare_metrics(
        metrics_original, 
        metrics_smote, 
        "SMOTE"
    )
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        original_vals = [
            metrics_original['Accuracy'],
            metrics_original['Precision'],
            metrics_original['Recall'],
            metrics_original['F1-Score']
        ]
        smote_vals = [
            metrics_smote['Accuracy'],
            metrics_smote['Precision'],
            metrics_smote['Recall'],
            metrics_smote['F1-Score']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_vals, width, label='Original', color='#1f77b4')
        bars2 = ax.bar(x + width/2, smote_vals, width, label='SMOTE', color='#2ca02c')
        
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison: Original vs SMOTE')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        improvements = [
            (metrics_smote['Accuracy'] - metrics_original['Accuracy']) * 100,
            (metrics_smote['Precision'] - metrics_original['Precision']) * 100,
            (metrics_smote['Recall'] - metrics_original['Recall']) * 100,
            (metrics_smote['F1-Score'] - metrics_original['F1-Score']) * 100
        ]
        
        colors = ['#2ca02c' if x >= 0 else '#d62728' for x in improvements]
        bars = ax.barh(metrics_names, improvements, color=colors)
        
        ax.set_xlabel('Improvement (%)')
        ax.set_title('Performance Improvement with SMOTE')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, improvements):
            ax.text(val, bar.get_y() + bar.get_height()/2.,
                   f'{val:.2f}%', ha='left' if val >= 0 else 'right',
                   va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Key Insights
    st.subheader("📝 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recall_improvement = (metrics_smote['Recall'] - metrics_original['Recall']) * 100
        st.info(f"**Recall Improvement:** {recall_improvement:+.2f}%\n\n"
               f"SMOTE significantly improves minority class detection!")
    
    with col2:
        f1_improvement = (metrics_smote['F1-Score'] - metrics_original['F1-Score']) * 100
        st.info(f"**F1-Score Improvement:** {f1_improvement:+.2f}%\n\n"
               f"Better balance between precision and recall!")
    
    with col3:
        samples_added = smote_info['Samples Added']
        st.info(f"**Samples Generated:** {samples_added}\n\n"
               f"SMOTE created synthetic minority samples!")
    
    # ======== SECTION 6: GAN COMPARISON (Optional) ========
    if st.session_state.compare_with_gan:
        st.header("🎨 Section 6: Comparison with GAN-Based Balancing")
        
        with st.spinner("Training GAN model (this may take a moment)..."):
            model_gan, y_pred_gan, y_pred_proba_gan, metrics_gan = apply_gan_to_data(
                st.session_state.X_train, 
                st.session_state.y_train,
                st.session_state.X_test,
                st.session_state.y_test,
                st.session_state.model_type_key
            )
        
        # GAN metrics display
        st.subheader("GAN Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics_gan['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics_gan['Precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics_gan['Recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics_gan['F1-Score']:.4f}")
        
        # Three-way comparison
        st.subheader("Three-Way Performance Comparison")
        
        comparison_three_way = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Original": [
                f"{metrics_original['Accuracy']:.4f}",
                f"{metrics_original['Precision']:.4f}",
                f"{metrics_original['Recall']:.4f}",
                f"{metrics_original['F1-Score']:.4f}"
            ],
            "SMOTE": [
                f"{metrics_smote['Accuracy']:.4f}",
                f"{metrics_smote['Precision']:.4f}",
                f"{metrics_smote['Recall']:.4f}",
                f"{metrics_smote['F1-Score']:.4f}"
            ],
            "GAN": [
                f"{metrics_gan['Accuracy']:.4f}",
                f"{metrics_gan['Precision']:.4f}",
                f"{metrics_gan['Recall']:.4f}",
                f"{metrics_gan['F1-Score']:.4f}"
            ]
        })
        
        st.dataframe(comparison_three_way, use_container_width=True, hide_index=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            original_vals = [
                metrics_original['Accuracy'],
                metrics_original['Precision'],
                metrics_original['Recall'],
                metrics_original['F1-Score']
            ]
            smote_vals = [
                metrics_smote['Accuracy'],
                metrics_smote['Precision'],
                metrics_smote['Recall'],
                metrics_smote['F1-Score']
            ]
            gan_vals = [
                metrics_gan['Accuracy'],
                metrics_gan['Precision'],
                metrics_gan['Recall'],
                metrics_gan['F1-Score']
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.25
            
            ax.bar(x - width, original_vals, width, label='Original', color='#1f77b4')
            ax.bar(x, smote_vals, width, label='SMOTE', color='#2ca02c')
            ax.bar(x + width, gan_vals, width, label='GAN', color='#d62728')
            
            ax.set_ylabel('Score')
            ax.set_title('Metrics Comparison: Original vs SMOTE vs GAN')
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
                (metrics_smote['Accuracy'] - metrics_original['Accuracy']) * 100,
                (metrics_smote['Precision'] - metrics_original['Precision']) * 100,
                (metrics_smote['Recall'] - metrics_original['Recall']) * 100,
                (metrics_smote['F1-Score'] - metrics_original['F1-Score']) * 100
            ]
            
            gan_improvements = [
                (metrics_gan['Accuracy'] - metrics_original['Accuracy']) * 100,
                (metrics_gan['Precision'] - metrics_original['Precision']) * 100,
                (metrics_gan['Recall'] - metrics_original['Recall']) * 100,
                (metrics_gan['F1-Score'] - metrics_original['F1-Score']) * 100
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
        
        # Final Conclusion
        st.header("🎯 Final Conclusion: SMOTE vs GAN")
        
        # Determine winner based on F1-score
        smote_f1 = metrics_smote['F1-Score']
        gan_f1 = metrics_gan['F1-Score']
        original_f1 = metrics_original['F1-Score']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SMOTE Advantages:")
            st.markdown("""
            - ✅ **Fast Training:** SMOTE is computationally efficient
            - ✅ **Deterministic:** Creates oversamples based on interpolation
            - ✅ **Interpretable:** Clear methodology for synthetic sample generation
            - ✅ **Stable:** Consistent results across runs
            - ✅ **Low Risk:** Less prone to overfitting
            """)
        
        with col2:
            st.subheader("GAN Advantages:")
            st.markdown("""
            - ✅ **High Quality:** Creates realistic complex synthetic samples
            - ✅ **Flexible:** Can learn complex data distributions
            - ✅ **Modern:** Uses deep learning for sample generation
            - ✅ **Scalable:** Works well with high-dimensional data
            - ✅ **Creative:** Generates diverse samples
            """)
        
        st.markdown("---")
        
        # Recommendation with robust percentage calculation
        if smote_f1 > gan_f1:
            recommendation = "SMOTE"
            absolute_diff = (smote_f1 - gan_f1) * 100  # Convert to percentage points
            relative_pct = ((smote_f1 - gan_f1) / gan_f1) * 100
            # Cap relative percentage at 100% for display
            relative_pct_capped = min(relative_pct, 100)
            better_text = "in favor of SMOTE"
        else:
            recommendation = "GAN"
            absolute_diff = (gan_f1 - smote_f1) * 100  # Convert to percentage points
            relative_pct = ((gan_f1 - smote_f1) / smote_f1) * 100
            # Cap relative percentage at 100% for display
            relative_pct_capped = min(relative_pct, 100)
            better_text = "in favor of GAN"
        
        # Calculate improvements over original
        smote_improvement = ((smote_f1 - original_f1) / original_f1) * 100
        gan_improvement = ((gan_f1 - original_f1) / original_f1) * 100
        
        # Cap improvements at 100% for display
        smote_improvement_capped = min(smote_improvement, 100)
        gan_improvement_capped = min(gan_improvement, 100)
        
        st.success(f"""
        ### 📌 Recommendation: **{recommendation}** is the better choice
        
        **F1-Score Comparison:**
        - Original Model: **{original_f1:.4f}**
        - SMOTE: **{smote_f1:.4f}** (improvement: {smote_improvement_capped:.1f}%)
        - GAN: **{gan_f1:.4f}** (improvement: {gan_improvement_capped:.1f}%)
        
        **Performance Margin:** {absolute_diff:.2f} percentage points {better_text}
        
        """)
        
        if recommendation == "SMOTE":
            st.info("""
            SMOTE is the better choice for this use case because:
            
            1. **Efficiency:** Much faster to train and deploy
            2. **Stability:** Produces consistent, reproducible results
            3. **Performance:** Achieves comparable or better F1-score
            4. **Maintainability:** Easier to understand and maintain
            5. **Resource Usage:** Lower computational requirements
            
            **When to consider GAN instead:**
            - When you have very high-dimensional data (>100 features)
            - When you need maximum sample quality and realism
            - When computational resources are not a constraint
            - When working with complex, non-linear data distributions
            """)
        else:
            st.info(f"""
            GAN is the better choice for this use case because:
            
            1. **Superior Performance:** {absolute_diff:.2f} percentage points better F1-score
            2. **Quality:** Generates more sophisticated synthetic samples
            3. **Adaptability:** Better captures complex data patterns
            4. **Consistency:** More balanced learning of data distribution
            
            **However, consider SMOTE if:**
            - Training time is critical for your application
            - You need guaranteed reproducibility
            - Computational resources are limited
            - F1-score difference is marginal (<2%)
            """)

else:
    st.info("👈 Configure your simulation settings above and click the 'Run Simulation' button to start!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>SMOTE Virtual Lab</strong> | Demonstrating Class Imbalance Handling Techniques</p>
    <p style='font-size: 0.9em; color: gray;'>Made with Streamlit | SMOTE vs GAN Comparison</p>
</div>
""", unsafe_allow_html=True)
