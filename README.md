# ✈️ Flight Delay Prediction Dashboard

Interactive Streamlit dashboard to analyze US flight delays and predict whether a flight is **Delayed (1)** or **On-Time (0)**.

I built this as an end-to-end ML project:
- Data preprocessing & feature engineering in Python
- Training multiple models (Logistic Regression, Random Forest, XGBoost)
- Model comparison & feature importance analysis
- Interactive dashboard for analysis + prediction

---

## 🧠 Models Used

I trained and compared two main models:

- **Random Forest Classifier**
- **XGBoost Classifier**  ✅ (best performer)

Both models are trained on engineered features like:
- `DEP_DELAY`, `ARR_DELAY`, `DISTANCE`
- `DEP_HOUR`, `DAY_OF_WEEK`, `MONTH`
- Encoded airline, origin, destination
- Historical average delay features (origin, carrier, route)

---

## 📂 Repository Structure

```text
flight_delay_dashboard/
├── app.py                 # Streamlit dashboard code
├── requirements.txt       # Python dependencies
├── xgb_feature_importance.csv  # Top features from XGBoost
├── xgboost_model.pkl      # Saved XGBoost model (binary)
├── data/                  # Placeholder for local CSVs (see below)
└── models/                # (Optional) local models folder
