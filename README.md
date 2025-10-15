# 📱 Number Pricing Pipeline

> **Standalone ML package for Thai phone number price prediction**
> Clean architecture • Environment-agnostic • Production-ready

---

## 🎯 Overview

This is a **complete, independent ML package** for predicting Thai phone number prices based on numerical patterns, cultural significance, and market demand.

**Key Features:**
- ✅ **Fully isolated**: No dependencies on legacy codebases
- ✅ **Config-driven**: All settings centralized in `config.py`
- ✅ **Environment detection**: Auto-detects Local, Colab, Kaggle, Paperspace
- ✅ **Data leakage prevention**: Strict validation and pipeline design
- ✅ **Production-ready**: Clean architecture, logging, error handling

**Created by:** Codex AI (2025)
**Dataset:** `numberdata.csv` (6,100+ Thai phone numbers with prices)

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/number-pricing.git
cd number-pricing
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset

Place your dataset at:
```
number_pricing/
└── data/
    └── raw/
        └── numberdata.csv  # Required columns: phone_number, price
```

**Dataset format:**
```csv
phone_number,price
0812345678,500
0899999999,50000
```

---

## 🚀 Quick Start

### Training

```bash
# Train model with default settings
python -m number_pricing.scripts.train

# Artifacts saved to: number_pricing_artifacts/
```

### Prediction

```bash
# Predict single or multiple numbers
python -m number_pricing.scripts.predict --numbers 0812345678 0991234567

# Or from CSV file
python -m number_pricing.scripts.predict --input-file numbers.csv
```

---

## 📂 Project Structure

```
number_pricing/
├── config.py              # 🎛️ Centralized configuration (SINGLE SOURCE OF TRUTH)
├── data/                  # 📊 Dataset loading & validation
│   ├── __init__.py
│   └── dataset_loader.py
├── features/              # 🔧 Feature engineering
│   ├── __init__.py
│   └── feature_extractor.py
├── models/                # 🤖 Model factory
│   ├── __init__.py
│   └── model_factory.py
├── pipelines/             # ⚙️ Training & prediction orchestration
│   ├── __init__.py
│   ├── training_pipeline.py
│   └── prediction_pipeline.py
├── evaluation/            # 📈 Metrics & evaluation
│   ├── __init__.py
│   └── metrics.py
├── utils/                 # 🛠️ Logging, I/O, validation
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── io.py
│   └── validation.py
├── scripts/               # 🖥️ CLI entry points
│   ├── __init__.py
│   ├── train.py          # Training script
│   └── predict.py        # Prediction script
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All settings are in **`config.py`** (single source of truth):

### Key Sections

#### 1. **Paths** (Auto-detection)
```python
paths:
  - dataset_location: Auto-detects environment (local/Colab/Kaggle/Paperspace)
  - artifact_destination: Output directory for models/reports
```

#### 2. **Data**
```python
data:
  - phone_column: "phone_number"
  - price_column: "price"
  - validation_rules: Length checks, deduplication
```

#### 3. **Features**
```python
features:
  - premium_patterns: 8888, 9999, 168, etc.
  - penalty_patterns: Unlucky numbers
  - positional_groupings: ABC position scoring
  - aggregations: Sum, mean, entropy, etc.
```

#### 4. **Model**
```python
model:
  - estimator_name: "random_forest" (or "xgboost", "lightgbm", etc.)
  - hyperparameters: Model-specific params
  - cv_strategy: Cross-validation settings
  - target_transform: Log-scaling for prices
```

### Environment Variables (Optional)

```bash
export NUMBER_PRICING_ENV=colab              # Force environment
export NUMBER_PRICING_DATASET_FILENAME=my_numbers.csv  # Custom dataset
```

---

## 🔒 Data Leakage Safeguards

1. **Validation first**: Remove malformed/duplicate numbers BEFORE split
2. **Pipeline design**: Feature extraction inside sklearn pipeline
3. **Target transform**: Log-scaling via `TransformedTargetRegressor`
4. **No leakage**: Train/test split on raw numbers, features created per-fold

---

## 🎨 Extending the Package

### Add New Features

Edit `FeatureSettings` in `config.py`:

```python
# config.py
class FeatureSettings:
    premium_patterns = {
        '8888': 10.0,
        '9999': 9.5,
        '1688': 8.0,  # Add new pattern
    }
```

No other file needs changes! ✨

### Swap Models

Change `estimator_name` in `config.py`:

```python
# config.py
class ModelSettings:
    estimator_name = "xgboost"  # Or: random_forest, lightgbm, catboost

    # Tune hyperparameters
    hyperparameters = {
        'n_estimators': 500,
        'max_depth': 10,
        ...
    }
```

All registered in `models/model_factory.py`.

---

## 📊 Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Local** | ✅ Supported | Default environment |
| **Google Colab** | ✅ Supported | Auto-detects `/content/` |
| **Kaggle** | ✅ Supported | Auto-detects `/kaggle/` |
| **Paperspace** | ✅ Supported | Auto-detects `/notebooks/` |

Environment detection is automatic via `config.py`.

---

## 🧪 Testing (Coming Soon)

```bash
# Run tests (when implemented)
pytest tests/

# Coverage report
pytest --cov=number_pricing tests/
```

---

## 📝 Example Usage

### Complete Workflow

```python
# 1. Train model
python -m number_pricing.scripts.train

# Output:
# ✅ Model trained: number_pricing_artifacts/model.pkl
# ✅ Metrics: R² = 0.85, MAE = 1234.56
# ✅ Report: number_pricing_artifacts/training_report.json

# 2. Make predictions
python -m number_pricing.scripts.predict --numbers 0899999999 0812345678

# Output:
# 0899999999 → ฿48,500
# 0812345678 → ฿1,200
```

---

## 🤝 Contributing

This is an **independent project** created by Codex AI.

**Guidelines:**
1. All config changes go in `config.py` only
2. Follow existing architecture (no hard-coded params)
3. Add tests for new features
4. Update this README for major changes

---

## 📄 License

MIT License (or specify your license)

---

## 🔗 Related Projects

**Legacy ML Pipeline** (original implementation):
[github.com/YOUR_USERNAME/number-ML](https://github.com/YOUR_USERNAME/number-ML)

> **Note**: This package (`number-pricing`) is a **complete rewrite** using the same dataset but independent codebase.

---

## 🙏 Acknowledgments

- **Dataset**: Thai phone number pricing data
- **Created by**: Codex AI (2025)
- **Inspired by**: ML Phone Number Price Prediction (legacy project)

---

## 📞 Support

For issues or questions:
1. Check `PAPERSPACE_TRAINING_GUIDE.md` for platform-specific tips
2. Review `config.py` for configuration options
3. Open an issue on GitHub

---

**Built with ❤️ by Codex AI**
