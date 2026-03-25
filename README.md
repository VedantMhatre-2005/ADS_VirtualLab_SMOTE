# 🧪 SMOTE Virtual Lab: Imbalanced Dataset Handling

A comprehensive Streamlit-based virtual laboratory demonstrating the **SMOTE (Synthetic Minority Over-sampling Technique)** for handling imbalanced datasets in machine learning, with optional GAN comparison.

## 📋 Overview

This interactive virtual lab provides a complete pipeline for:
1. **Loading imbalanced datasets** - Choose from 4 different imbalanced datasets
2. **Exploratory Data Analysis (EDA)** - Visualize class imbalance issues
3. **Data Splitting** - Proper train-test stratification
4. **Model Training** - Train on original imbalanced data
5. **SMOTE Application** - Apply SMOTE to balance training data
6. **Retraining** - Train model on balanced data
7. **Performance Comparison** - Compare metrics before/after SMOTE
8. **GAN Comparison** - Optional comparison with GAN-based balancing

## 🎯 Features

### Four Imbalanced Datasets
1. **Credit Card Fraud** (2% fraud rate)
2. **Disease Detection** (5% disease rate)
3. **Network Intrusion** (4% intrusion rate)
4. **Rare Event Prediction** (8% rare event rate)

### Classification Models
- Random Forest
- Logistic Regression

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Visual comparisons and improvement analysis

### Balancing Techniques
- **SMOTE** - Synthetic Minority Over-sampling Technique
- **GAN** - Generative Adversarial Networks (optional, computationally intensive)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd ADS_VirtualLab
```

2. **Create a virtual environment (recommended):**
```bash
# Using venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

1. **Select Dataset**: Choose one of the four imbalanced datasets from the sidebar
2. **Choose Model**: Select either Random Forest or Logistic Regression
3. **Enable GAN Comparison** (Optional): Check to compare with GAN-based balancing (slower)
4. **Run Analysis**: Click the "Run Analysis" button to execute the pipeline

The app will:
- Display dataset information and class distribution
- Train a model on original imbalanced data
- Apply SMOTE to balance the training data
- Retrain the model on balanced data
- Show detailed performance metrics and comparisons
- Optionally compare with GAN results

## 📊 Understanding the Results

### Key Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: How many positive predictions were correct
- **Recall**: How many actual positives were found (important for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall (best metric for imbalanced data)
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Expected Improvements with SMOTE

Typically you should see:
- **↑ Recall** - Better detection of minority class
- **↑ F1-Score** - Better overall balance
- **Slight changes in Accuracy** - May increase or decrease
- **↑ Precision** - Often improves due to better learning

## 🎨 SMOTE vs GAN Comparison

### SMOTE (Synthetic Minority Over-sampling Technique)
**Pros:**
- Fast and efficient
- Deterministic results
- Easy to understand
- Low computational cost
- Less prone to overfitting

**Cons:**
- Creates linear interpolations
- May create unrealistic samples
- Limited to simple patterns

### GAN (Generative Adversarial Networks)
**Pros:**
- Creates realistic, complex samples
- Can learn sophisticated distributions
- Better for high-dimensional data
- More flexible sample generation

**Cons:**
- Computationally intensive
- Longer training time
- More complex to tune
- Risk of mode collapse
- Requires more data

## 📁 Project Structure

```
ADS_VirtualLab/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── utils/
    ├── data_loader.py    # Dataset loading and preparation
    ├── models.py         # Model training and evaluation
    ├── smote_handler.py  # SMOTE implementation
    └── gan_handler.py    # GAN-based balancing
```

## 🛠️ Customization

### Adding New Datasets

Edit `utils/data_loader.py` and add your dataset in the `load_imbalanced_dataset()` function:

```python
elif dataset_name == "Your Dataset":
    X, y = make_classification(...)
    # Or load your own data
```

### Changing Model Parameters

Edit `utils/models.py` in the `ClassificationModel.__init__()` method:

```python
self.model = RandomForestClassifier(
    n_estimators=200,  # Change this
    max_depth=15,      # Change this
    ...
)
```

### Adjusting SMOTE Parameters

Edit `app.py` in Section 3:

```python
smote_handler = SMOTEHandler(
    k_neighbors=7,  # Change number of neighbors
    random_state=42
)
```

## 🔬 Technical Details

### SMOTE Algorithm
SMOTE works by:
1. Selecting a random minority class sample
2. Finding its K nearest minority class neighbors
3. Creating synthetic samples along the line connecting to neighbors
4. Repeating until minority class is balanced

### Class Imbalance Problem
- Models trained on imbalanced data are biased toward majority class
- Accuracy becomes a misleading metric
- Recall becomes more important than accuracy
- SMOTE helps models learn minority class better

### GAN Balancing
- Generator learns to create realistic minority class samples
- Discriminator learns to distinguish real from synthetic samples
- Through adversarial training, both improve
- Generated samples are more complex than SMOTE's linear interpolations

## 📈 Performance Tips

1. **For faster results**: Disable GAN comparison initially
2. **For better SMOTE results**: Ensure data is properly scaled (done automatically)
3. **For GAN training**: Reduce epochs if training is too slow
4. **For reproducibility**: Keep random_state values consistent

## ⚠️ Limitations

- GAN training may fail if TensorFlow is not properly installed
- Very large datasets may be slow with the current implementation
- GAN comparison is computationally intensive; expect 1-3 minutes
- Synthetic data quality depends on original data quality

## 🤝 Contributing

Feel free to extend this lab with:
- More datasets
- Additional balancing techniques (ADASYN, BorderlineSMOTE, etc.)
- More classification models (SVM, XGBoost, Neural Networks)
- Additional visualizations
- Hyperparameter tuning interface

## 📚 References

- **SMOTE Paper**: Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- **Class Imbalance**: He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data"
- **GANs**: Goodfellow, I., et al. (2014). "Generative Adversarial Nets"
- **Scikit-learn**: https://scikit-learn.org/
- **Imbalanced-learn**: https://imbalanced-learn.org/

## 📝 License

This project is provided as-is for educational purposes.

## ✉️ Questions & Support

For issues or questions:
1. Check the troubleshooting section below
2. Review the code comments in the utils modules
3. Ensure all dependencies are properly installed

### Troubleshooting

**Module not found error:**
```bash
pip install -r requirements.txt
```

**TensorFlow issues:**
```bash
pip install tensorflow --upgrade
```

**Streamlit not starting:**
```bash
streamlit run app.py --logger.level=debug
```

---

**Enjoy exploring SMOTE and class imbalance solutions!** 🚀
