# ✈️ Flight Delay Prediction Dashboard
🚀 **Live Demo:** Deployment in progress — run locally using instructions below.

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
## 📂 Data & Model Files (Google Drive)

Because the full dataset and trained models are large, I store them on Google Drive.

🔗 **All files (CSV + models) are here:**  
https://drive.google.com/drive/folders/1YoH2VIrg42jhH9Xpk3wY6ss_MCbrce8R?usp=drive_link

Download these and place them like this if you want to run the project locally:

```bash
flight_delay_dashboard/
├── app.py
├── requirements.txt
├── data/
│   ├── cleaned_flights.csv
│   ├── flights_sample_3m.csv
│   ├── processed_flights.csv
│   └── dictionary.html


## 🖥️ How to Run This Project Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/SparshikaDinesh/flight_delay_dashboard.git
   cd flight_delay_dashboard

├── models/
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt


   ## 📬 Contact

If you want to collaborate or hire me for Data/ML projects:

**Sparshika Ajmaan Dinesh Kumar**  
📧 sparshikaajmaan707@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/sparshikaajmaan/


3. Run the Streamlit app:
   ```bash
   streamlit run app.py

