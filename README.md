# Vehicle Price Regression - Datium Dataset

A comprehensive machine learning project for predicting vehicle sold amounts using traditional and advanced regression algorithms with MLflow experiment tracking.

## 📋 Project Overview

This project implements a regression model to predict `Sold_Amount` for vehicles using the Datium dataset. It includes:

- **Exploratory Data Analysis (EDA)** - Comprehensive data exploration and visualization
- **Data Preprocessing** - Feature engineering and data cleaning
- **Multiple ML Models** - Comparison of various regression algorithms
- **Hyperparameter Tuning** - Optimization using RandomizedSearchCV
- **Experiment Tracking** - MLflow integration for model versioning
- **Model Evaluation** - Residual analysis and feature importance

### Optional Advanced Feature:
- **FT-Transformer (Deep Learning)** - Optional implementation available for advanced tabular data modeling

## 🗂️ Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Core Python dependencies
├── requirements-ft-transformer.txt    # Optional FT-Transformer dependencies
├── .gitignore                         # Git ignore patterns
├── vehicle_price_regression.ipynb     # Main Jupyter notebook with complete analysis
├── ft_transformer_implementation.py   # FT-Transformer Python implementation
├── FT_TRANSFORMER_GUIDE.md           # Guide for FT-Transformer integration
├── ft_transformer_cells.md           # Notebook cells for FT-Transformer
├── DatiumTrain.rpt                   # Training dataset
├── DatiumTest.rpt                    # Test dataset
├── car_desc.csv                      # Car description data
├── best_model.joblib                 # Saved best model
├── predictions.csv                   # Model predictions on test set
└── mlruns/                           # MLflow experiment tracking logs
    ├── 0/                            # Default experiment
    └── 810653522730616722/           # Vehicle price regression experiments
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Test
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   # Option A: Install from requirements file (recommended)
   pip install -r requirements.txt
   
   # Option B: Install manually
   pip install pandas numpy matplotlib seaborn scikit-learn
   pip install xgboost lightgbm category_encoders
   pip install mlflow joblib jupyter
   ```

4. **[Optional] Install FT-Transformer dependencies** (only if you want to use the advanced deep learning model):
   ```bash
   # Option A: Install from requirements file
   pip install -r requirements-ft-transformer.txt
   
   # Option B: Install manually
   # For CPU version:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install rtdl
   
   # For GPU support (if you have CUDA):
   # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # pip install rtdl
   ```

### Running the Analysis

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook:**
   - Navigate to `vehicle_price_regression.ipynb`
   - Run all cells sequentially: `Cell > Run All`

3. **View MLflow tracking UI:**
   ```bash
   mlflow ui
   ```
   - Open browser to `http://localhost:5000`
   - Explore experiment runs and compare models

## 📊 Models Implemented

The project compares the following regression models:

### Traditional ML Models:
- **Linear Regression** - Baseline model
- **Ridge Regression** - L2 regularization
- **Lasso Regression** - L1 regularization
- **Elastic Net** - Combined L1 and L2 regularization
- **Random Forest** - Ensemble tree-based model
- **Gradient Boosting** - Boosted decision trees

### Advanced Ensemble Methods:
- **XGBoost** - Extreme Gradient Boosting
- **LightGBM** - Light Gradient Boosting Machine

## 🎯 Key Features

### Data Processing
- Intelligent null value handling
- Feature engineering (Create Date, First Registration Date, Model Year)
- Categorical encoding (Label Encoding, Target Encoding)
- Feature standardization with StandardScaler

### Model Evaluation
- Cross-validation with KFold
- Multiple metrics: RMSE, MAE, R²
- Residual analysis
- Feature importance analysis
- Permutation importance

### Experiment Tracking
- MLflow integration for all experiments
- Model versioning and comparison
- Hyperparameter logging
- Artifact storage (models, plots)

## 📈 Results

The best performing model is automatically saved as `best_model.joblib` and can be loaded for predictions:

```python
import joblib
import pandas as pd

# Load the best model
model = joblib.load('best_model.joblib')

# Load test data
test_data = pd.read_csv('DatiumTest.rpt', sep='|')

# Make predictions
predictions = model.predict(test_data)
```

## 🔍 MLflow Experiment Tracking

All experiments are tracked using MLflow. To view:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to:
- Compare model performance
- View hyperparameters
- Download trained models
- Analyze training metrics
- Export results

## 📖 Documentation

- **[FT_TRANSFORMER_GUIDE.md](FT_TRANSFORMER_GUIDE.md)** - Comprehensive guide for FT-Transformer integration
- **[ft_transformer_cells.md](ft_transformer_cells.md)** - Ready-to-use notebook cells
- **Main Notebook** - Detailed comments and explanations in each cell

## 🚀 Optional: FT-Transformer Deep Learning Model

For advanced users interested in exploring deep learning approaches, this repository includes an optional **FT-Transformer** (Feature Tokenizer Transformer) implementation.

### What is FT-Transformer?

FT-Transformer is a state-of-the-art deep learning architecture designed specifically for tabular data, offering several advantages:

- **No manual feature scaling required** - Works with raw numerical features
- **Native categorical handling** - Embedding-based categorical encoding
- **Attention mechanism** - Captures complex feature interactions
- **State-of-the-art performance** - Competitive with or better than gradient boosting on many tasks

### How to Use FT-Transformer

**Option 1: Standalone Python Script**
```bash
python ft_transformer_implementation.py
```

**Option 2: Integrate into Notebook**

Follow the step-by-step guide in [FT_TRANSFORMER_GUIDE.md](FT_TRANSFORMER_GUIDE.md) to add FT-Transformer cells to your notebook.

**Prerequisites:**
- PyTorch installed (see optional installation step above)
- RTDL library (`pip install rtdl`)

### When to Use FT-Transformer?

Consider using FT-Transformer when:
- You have sufficient data (typically >10,000 samples)
- You want to explore deep learning alternatives to gradient boosting
- You have GPU resources available for faster training
- You're interested in capturing complex non-linear feature interactions

## 🛠️ Technical Details

### Core ML Pipeline

The main notebook implements a complete machine learning pipeline:

1. **Data Loading & Processing:**
   - Pipe-delimited file parsing
   - Feature type identification
   - Missing value handling

2. **Feature Engineering:**
   - Date feature extraction (Create Date, Registration Date)
   - Model year derivation
   - Categorical encoding strategies

3. **Model Training:**
   - Cross-validation for robust evaluation
   - Hyperparameter tuning with RandomizedSearchCV
   - Multiple algorithm comparison

4. **Model Selection:**
   - Performance metrics comparison (RMSE, MAE, R²)
   - Best model selection and persistence

### FT-Transformer Architecture (Optional)

For users implementing the optional FT-Transformer, the architecture includes:

1. **Feature Tokenization:**
   - Numerical features → Linear projection to embeddings
   - Categorical features → Embedding lookup tables

2. **Transformer Layers:**
   - Multi-head self-attention mechanism
   - Feed-forward neural networks
   - LayerNorm and residual connections

3. **Benefits:**
   - Works with raw features (no manual scaling)
   - Captures complex feature interactions
   - State-of-the-art performance on tabular data

### Package Requirements

**Core Requirements:**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
category-encoders>=2.3.0
mlflow>=1.20.0
joblib>=1.1.0
jupyter>=1.0.0
```

**Optional (for FT-Transformer only):**
```
torch>=2.0.0
rtdl>=0.0.13
```

## 🔐 Security Considerations

This project follows security best practices as per Accenture standards:

- **Input Validation:** All input data is validated before processing
- **Data Integrity:** Checksums and validation for data loading
- **Secure Dependencies:** Regular updates to address vulnerabilities
- **No Hardcoded Credentials:** All sensitive data externalized
- **Secure Code Practices:** Following Secure Vibe Coding Guidelines

## ♻️ Resource Efficiency

Following green software principles:

- **Efficient Algorithms:** Optimized for computational efficiency
- **Memory Management:** Proper cleanup and resource management
- **Model Optimization:** Balanced accuracy vs. computational cost
- **Batch Processing:** Efficient data loading and processing

## 🤝 Contributing

When contributing to this project:

1. Follow the coding standards in the main notebook
2. Add documentation for new features
3. Update this README with any new dependencies
4. Test all changes before committing
5. Use meaningful commit messages

## 📝 License

This project is for Accenture internal use. All rights reserved.

## 👥 Contact

For questions or support, please contact the project maintainer.

---

**Generated by GitHub Copilot**  
Last updated: February 16, 2026
