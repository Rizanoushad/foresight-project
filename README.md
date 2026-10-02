# 📦 FORESIGHT — AI-Powered Demand & Inventory Intelligence

FORESIGHT is a machine learning project that helps businesses understand product demand and make better inventory decisions.

The project takes historical retail sales data, forecasts future demand for different products, identifies possible inventory risks, and shows the results through an interactive Streamlit dashboard.

##  What Problem Does It Solve?

Managing inventory can be difficult:

- Too little stock can lead to stockouts.
- Too much stock can lead to excess inventory.
- Changing customer demand makes planning harder.

FORESIGHT uses historical sales patterns to help identify these situations and provide simple inventory recommendations.

## 🔄 How It Works


Retail Sales Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Demand Forecasting
       ↓
Model Evaluation
       ↓
Inventory Risk Analysis
       ↓
Recommendations
       ↓
Streamlit Dashboard


##  Machine Learning

The project uses **HistGradientBoostingRegressor** to predict daily demand.

The model uses features such as:

- Previous demand (1, 7, 14 and 28 days)
- Rolling demand averages
- Demand variability
- Day of week
- Month and seasonality

The model achieved:

| Metric | Result |
|---|---:|
| Test MAE | **7.67** |
| Test RMSE | **33.23** |
| Baseline MAE | **9.82** |

The model improved MAE by approximately **21.8%** compared with the previous-day baseline.

##  Dashboard

The Streamlit dashboard allows you to:

- View overall demand information
- Select and analyze individual products
- Compare actual vs predicted demand
- View inventory risk
- Adjust forecast and inventory assumptions
- Get inventory recommendations
- Explore model feature importance

##  Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Streamlit**
- **Git & GitHub**

## 📁 Project Structure

```text
FORESIGHT/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── src/
│   ├── preprocessing/
│   ├── eda/
│   ├── visualization/
│   ├── models/
│   ├── inventory/
│   └── dashboard/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ How to Run

### 1. Create and activate the environment

```bash
conda create -n foresight python=3.11
conda activate foresight
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the project pipeline

```bash
python src/preprocessing/clean_data.py
python src/preprocessing/prepare_demand.py
python src/eda/eda_analysis.py
python src/visualization/visualize.py
python src/models/train_model.py
python src/models/model_interpretation.py
python src/inventory/inventory_risk.py
```

### 4. Start the dashboard

```bash
streamlit run src/dashboard/app.py
```

Then open:

```text
http://localhost:8501
```

## ⚠️ Important Note

The dataset does not contain actual inventory levels, supplier lead times, or reorder points.

Because of this, the inventory risk analysis is **scenario-based**. The dashboard uses configurable assumptions such as forecast horizon, inventory coverage, and safety buffer.

## 📚 What I Learned

Through this project, I worked with:

- Data cleaning
- Exploratory data analysis
- Feature engineering
- Machine learning
- Demand forecasting
- Model evaluation
- Error analysis
- Inventory risk analysis
- Streamlit dashboard development

## 👨‍💻 Author

**Riza Noushad**

GitHub: `https://github.com/Rizanoushad`