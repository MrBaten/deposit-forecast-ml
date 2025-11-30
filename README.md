# Customer Deposit Forecasting System

## 📊 Project Overview

A comprehensive machine learning system for predicting next-day deposit amounts for customers on a betting platform based on their historical deposit behavior over the last 12 months.

**Business Goal**: Predict tomorrow's deposit amount for each customer to improve financial planning and understand customer behavior patterns.

## 🎯 Project Status

- ✅ **Phase 1**: Data Preparation & EDA (COMPLETED)
- ✅ **Phase 2**: Feature Engineering (COMPLETED)
- ⏳ **Phase 3**: Model Development (NEXT)
- ⏳ **Phase 4**: Model Evaluation
- ⏳ **Phase 5**: Production Pipeline

## 📁 Project Structure

```
customer_deposit_forecasting/
│
├── data/                           # Data files
│   ├── customer_deposits_raw.csv           # Raw synthetic data (1000 customers, 365 days)
│   ├── customer_deposits_preprocessed.csv  # Preprocessed data with features
│   ├── customer_statistics.csv             # Per-customer statistics
│   └── outliers_detected.csv               # Detected outliers
│
├── src/                            # Source code
│   ├── data_generator.py              # Synthetic data generation
│   ├── eda.py                         # Exploratory data analysis
│   ├── preprocessing.py               # Data preprocessing pipeline
│   └── run_phase1.py                  # Phase 1 execution script
│
├── notebooks/                      # Jupyter notebooks
│   └── phase1_data_preparation_eda.ipynb
│
├── visualizations/                 # Generated visualizations
│   ├── phase1_summary.png            # Overall EDA summary
│   └── sample_customer_patterns.png  # Sample customer time series
│
├── models/                         # Trained models (to be created)
├── outputs/                        # Model predictions (to be created)
├── tests/                          # Unit tests (to be created)
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
```bash
cd customer_deposit_forecasting
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running Phase 1

To run the complete Phase 1 workflow (data generation, EDA, and preprocessing):

```bash
cd src
python run_phase1.py
```

This will:
- Generate synthetic deposit data for 1000 customers
- Perform comprehensive exploratory data analysis
- Preprocess the data and create temporal features
- Generate visualizations
- Save all outputs to the appropriate directories

## 📊 Dataset Overview

### Synthetic Data Characteristics

**Dataset Size**:
- **Customers**: 1,000
- **Time Period**: 365 days (12 months)
- **Total Records**: 365,000
- **Total Deposit Volume**: $25.6M
- **Transaction Count**: ~122,000

### Customer Segments

The system simulates 7 distinct customer behavior segments:

| Segment | Proportion | Description |
|---------|-----------|-------------|
| **High Frequency Regular** | 15% | Daily depositors with consistent amounts (~$126 avg) |
| **Medium Frequency Stable** | 25% | Weekly depositors, most common segment (~$247 avg) |
| **Growing Users** | 20% | Increasing deposit amounts over time (~$258 avg) |
| **Declining Users** | 15% | Decreasing activity (~$112 avg) |
| **Sporadic High Value** | 10% | Rare but large deposits (~$596 avg) |
| **Weekend Warriors** | 10% | Primarily weekend depositors (~$354 avg) |
| **Inactive Declining** | 5% | Very low activity (~$13 avg) |

### Temporal Patterns

**Seasonality Effects**:
- ✅ Weekend boost (Saturdays and Sundays show higher volumes)
- ✅ Month-end spikes (days 25-31 show increased deposits)
- ✅ Weekly cycles visible in time series
- ✅ Realistic noise and outliers (~1,963 outliers detected)

**Daily Statistics**:
- Average Daily Volume: **$70,143**
- Max Daily Volume: **$118,467**
- Min Daily Volume: **$43,729**

## 🔍 Phase 1 Results

### Key Insights

1. **Customer Behavior is Diverse**: 7 distinct segments with vastly different patterns
2. **Strong Temporal Signals**: Clear weekend and month-end effects
3. **Data Quality**: Successfully handled outliers and missing values
4. **Feature Engineering**: Created 16 temporal features for modeling

### Generated Features

**Temporal Features** (12 added):
- `day_of_week`, `day_of_month`, `day_of_year`
- `week_of_year`, `month`
- `is_weekend`, `is_month_start`, `is_month_end`
- `dow_sin`, `dow_cos` (cyclical encoding for day of week)
- `month_sin`, `month_cos` (cyclical encoding for month)

### Visualizations

1. **phase1_summary.png**: Four-panel overview showing:
   - Deposit amount distribution
   - Daily deposit volume time series
   - Deposits by day of week
   - Customer segment distribution

2. **sample_customer_patterns.png**: Time series plots for 6 sample customers from different segments

## 🛠️ Technical Stack

### Core Libraries
- **pandas** (2.0+): Data manipulation
- **numpy** (1.24+): Numerical operations
- **matplotlib** (3.7+): Visualization
- **scikit-learn** (1.3+): Machine learning

### To Be Added in Later Phases
- **xgboost** & **lightgbm**: Gradient boosting models
- **tensorflow/keras**: Deep learning (LSTM)
- **statsmodels** & **pmdarima**: Time series analysis
- **pytest**: Unit testing

## 📈 Next Steps - Phase 2: Feature Engineering

The next phase will create rich features for forecasting:

### Lag Features
- Deposits from past 1, 3, 7, 14, 30 days
- Previous week/month same day deposits

### Rolling Statistics
- 7-day, 14-day, 30-day moving averages
- Rolling standard deviations (volatility)
- Rolling min/max values

### Customer Behavior Metrics
- Deposit frequency
- Average amount per transaction
- Trend direction (growing/declining)
- Volatility score
- Days since last deposit
- Total deposits to date

### Growth Metrics
- 7-day vs 30-day deposit change rate
- Week-over-week growth
- Month-over-month growth

### Train/Validation/Test Split
- **Train**: First 10 months (Dec 2023 - Sep 2024)
- **Validation**: Month 11 (October 2024)
- **Test**: Month 12 (November 2024)

## 🧪 Testing

Unit tests will be added in Phase 2 for all critical functions:

```bash
pytest tests/
```

## 📝 Code Quality

### Design Principles
- ✅ **Modular**: Separate modules for data generation, EDA, preprocessing, and modeling
- ✅ **Type Hints**: Functions include type annotations
- ✅ **Documentation**: Comprehensive docstrings and comments
- ✅ **Error Handling**: Robust error handling throughout
- ✅ **Logging**: Clear output for monitoring progress

### Best Practices
- Object-oriented design for preprocessing and modeling
- Configuration-based approach for hyperparameters
- Reproducible results (random seeds set)
- Production-ready code structure

## 📊 Data Dictionary

### Raw Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | int | Unique customer identifier (1-1000) |
| `date` | datetime | Transaction date |
| `deposit_amount` | float | Deposit amount in dollars (0 = no deposit) |
| `segment` | str | Customer behavior segment |

### Preprocessed Data Additional Fields

| Field | Type | Description |
|-------|------|-------------|
| `day_of_week` | int | Day of week (0=Monday, 6=Sunday) |
| `day_of_month` | int | Day of month (1-31) |
| `day_of_year` | int | Day of year (1-365) |
| `week_of_year` | int | Week number (1-52) |
| `month` | int | Month (1-12) |
| `is_weekend` | int | Binary flag (1=weekend, 0=weekday) |
| `is_month_start` | int | Binary flag (1=first 5 days) |
| `is_month_end` | int | Binary flag (1=last 7 days) |
| `dow_sin`, `dow_cos` | float | Cyclical encoding of day of week |
| `month_sin`, `month_cos` | float | Cyclical encoding of month |

## 🤝 Contributing

This is a demonstration project for time series forecasting. Future enhancements could include:
- Real-time prediction API
- Model monitoring and drift detection
- Automated retraining pipeline
- Customer churn prediction integration
- Revenue forecasting

## 📄 License

This project is for internal analytics and educational purposes.

## 👥 Contact

For questions or suggestions, please contact the Data Science Team.

---

**Last Updated**: December 1, 2024  
**Version**: 1.0.0 (Phase 1 Complete)
