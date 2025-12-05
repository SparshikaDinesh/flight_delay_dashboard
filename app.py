import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- CACHED LOADERS ----------

@st.cache_data
def load_data():
    df = pd.read_csv("processed_flights.csv")
    # Drop date (not used in model)
    if "FL_DATE" in df.columns:
        df = df.drop(columns=["FL_DATE"])

    # Categorical encoding (must match training!)
    categorical_features = ["AIRLINE_CODE", "ORIGIN", "DEST"]
    code_maps = {}

    for col in categorical_features:
        df[col] = df[col].astype("category")
        cats = list(df[col].cat.categories)
        code_maps[col] = {cat: i for i, cat in enumerate(cats)}
        df[col] = df[col].cat.codes

    # Fill missing
    df = df.fillna(0)

    # Feature columns for model
    target = "DELAY_LABEL"
    feature_cols = [c for c in df.columns if c != target]

    return df, feature_cols, code_maps


@st.cache_resource
def load_model():
    model = joblib.load("xgboost_model.pkl")
    return model


# ---------- MAIN APP ----------

df, feature_cols, code_maps = load_data()
model = load_model()

st.set_page_config(
    page_title="Flight Delay Analytics & Prediction",
    layout="wide",
    page_icon="✈️"
)

st.sidebar.title("✈️ Flight Delay Dashboard")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Airline Insights",
        "Airport Insights",
        "Route-Level Delays",
        "Delay Heatmap",
        "Feature Importance",
        "Delay Predictor"
    ]
)

st.sidebar.info(
    "Data: US DOT On-Time Performance (2019–2023) – sample processed dataset.\n\n"
    "Model: XGBoost binary classifier predicting if a flight is delayed."
)

# ---------- HELPER FUNCTIONS ----------

def label_delay_prob(p):
    if p < 0.2:
        return "Low"
    elif p < 0.5:
        return "Moderate"
    elif p < 0.8:
        return "High"
    else:
        return "Very High"


# ---------- PAGE: OVERVIEW ----------

if page == "Overview":
    st.title("✈️ Flight Delay Analytics & Prediction")

    col1, col2, col3 = st.columns(3)
    total_flights = len(df)
    delay_rate = df["DELAY_LABEL"].mean()
    avg_arr_delay = df["ARR_DELAY"].mean()

    col1.metric("Total Flights", f"{total_flights:,.0f}")
    col2.metric("Delay Ratio", f"{delay_rate*100:.1f}%")
    col3.metric("Avg Arrival Delay (mins)", f"{avg_arr_delay:.1f}")

    st.markdown("### Sample of Processed Data")
    st.dataframe(df.head(20))

    st.markdown("---")
    st.markdown("### Monthly Delay Trend")

    monthly = (
        df.groupby("MONTH")["DELAY_LABEL"]
        .mean()
        .reset_index()
        .sort_values("MONTH")
    )
    monthly["Delay %"] = monthly["DELAY_LABEL"] * 100
    monthly = monthly[["MONTH", "Delay %"]].set_index("MONTH")
    st.line_chart(monthly)

# ---------- PAGE: AIRLINE INSIGHTS ----------

elif page == "Airline Insights":
    st.title("🛩 Airline Delay Insights")

    airline_stats = (
        df.groupby("AIRLINE_CODE")
        .agg(
            flights=("DELAY_LABEL", "count"),
            delay_rate=("DELAY_LABEL", "mean"),
            avg_arr_delay=("ARR_DELAY", "mean"),
            avg_dep_delay=("DEP_DELAY", "mean")
        )
        .reset_index()
    )

    airline_stats["delay_rate_pct"] = airline_stats["delay_rate"] * 100

    st.subheader("Airline Performance Summary")
    st.dataframe(
        airline_stats.sort_values("delay_rate", ascending=False),
        use_container_width=True
    )

    st.markdown("### Top Airlines by Delay Rate")
    top_n = st.slider("Number of airlines to show", 3, 20, 10)

    chart_df = (
        airline_stats.sort_values("delay_rate", ascending=False)
        .head(top_n)
        .set_index("AIRLINE_CODE")[["delay_rate_pct"]]
    )
    st.bar_chart(chart_df)

# ---------- PAGE: AIRPORT INSIGHTS ----------

elif page == "Airport Insights":
    st.title("🛫 Airport Delay Insights (Origin Airports)")

    origin_stats = (
        df.groupby("ORIGIN")
        .agg(
            flights=("DELAY_LABEL", "count"),
            delay_rate=("DELAY_LABEL", "mean"),
            avg_arr_delay=("ARR_DELAY", "mean"),
            avg_dep_delay=("DEP_DELAY", "mean")
        )
        .reset_index()
    )

    min_flights = st.slider(
        "Minimum flights from an airport (to include in stats)",
        1000, 30000, 5000, step=1000
    )

    origin_stats_filtered = origin_stats[origin_stats["flights"] >= min_flights]
    origin_stats_filtered["delay_rate_pct"] = origin_stats_filtered["delay_rate"] * 100

    st.subheader("Airport Performance Summary")
    st.dataframe(
        origin_stats_filtered.sort_values("delay_rate", ascending=False),
        use_container_width=True
    )

    st.markdown("### Most Delayed Origin Airports (by Delay Rate)")
    top_n = st.slider("Top N airports", 3, 30, 15)
    chart_df = (
        origin_stats_filtered.sort_values("delay_rate", ascending=False)
        .head(top_n)
        .set_index("ORIGIN")[["delay_rate_pct"]]
    )
    st.bar_chart(chart_df)

# ---------- PAGE: ROUTE-LEVEL DELAYS ----------

elif page == "Route-Level Delays":
    st.title("🧭 Route-Level Delay Analysis")

    st.markdown("Average delay per **route (ORIGIN → DEST)**.")

    route_stats = (
        df.groupby(["ORIGIN", "DEST"])
        .agg(
            flights=("DELAY_LABEL", "count"),
            delay_rate=("DELAY_LABEL", "mean"),
            avg_arr_delay=("ARR_DELAY", "mean"),
        )
        .reset_index()
    )

    min_flights_route = st.slider(
        "Minimum flights per route",
        200, 10000, 1000, step=200
    )

    route_stats_filtered = route_stats[route_stats["flights"] >= min_flights_route]
    route_stats_filtered["delay_rate_pct"] = route_stats_filtered["delay_rate"] * 100

    col1, col2 = st.columns(2)

    with col1:
        origin_choice = st.selectbox(
            "Filter by Origin (optional)",
            options=["All"] + sorted(route_stats_filtered["ORIGIN"].unique().tolist())
        )

    with col2:
        dest_choice = st.selectbox(
            "Filter by Destination (optional)",
            options=["All"] + sorted(route_stats_filtered["DEST"].unique().tolist())
        )

    filtered = route_stats_filtered.copy()
    if origin_choice != "All":
        filtered = filtered[filtered["ORIGIN"] == origin_choice]
    if dest_choice != "All":
        filtered = filtered[filtered["DEST"] == dest_choice]

    st.subheader("Route Table")
    st.dataframe(
        filtered.sort_values("delay_rate", ascending=False),
        use_container_width=True
    )

    st.markdown("### Top Delayed Routes")
    top_n_routes = st.slider("Top N routes", 3, 30, 15)

    chart_df = (
        filtered.sort_values("delay_rate", ascending=False)
        .head(top_n_routes)
        .assign(route=lambda d: d["ORIGIN"] + " → " + d["DEST"])
        .set_index("route")[["delay_rate_pct"]]
    )
    st.bar_chart(chart_df)

# ---------- PAGE: DELAY HEATMAP ----------

elif page == "Delay Heatmap":
    st.title("🌡 Route Delay Heatmap (Origin vs Destination)")

    st.markdown(
        "Heatmap of average delay rate between top busy airports. "
        "Darker = higher delay rate."
    )

    # Choose top airports by volume
    airport_counts = (
        df.groupby("ORIGIN")["DELAY_LABEL"]
        .count()
        .sort_values(ascending=False)
    )

    top_k = st.slider("Number of top airports", 5, 25, 10)
    top_airports = airport_counts.head(top_k).index.tolist()

    heat_df = (
        df[df["ORIGIN"].isin(top_airports) & df["DEST"].isin(top_airports)]
        .groupby(["ORIGIN", "DEST"])["DELAY_LABEL"]
        .mean()
        .unstack()
    )

    st.dataframe(
        heat_df.style.background_gradient(cmap="Reds"),
        use_container_width=True
    )

# ---------- PAGE: FEATURE IMPORTANCE ----------

elif page == "Feature Importance":
    st.title("📊 Model Feature Importance")

    st.markdown(
        "Feature importance scores from the XGBoost model. "
        "Higher = more influence on delay prediction."
    )

    # Recompute importance based on current model & feature_cols
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)

    st.subheader("Importance Table")
    st.dataframe(fi_df, use_container_width=True)

    top_n = st.slider("Show top N features", 3, len(fi_df), min(15, len(fi_df)))

    st.markdown("### Top Features")
    chart_df = fi_df.head(top_n).set_index("feature")
    st.bar_chart(chart_df)

# ---------- PAGE: DELAY PREDICTOR ----------

elif page == "Delay Predictor":
    st.title("🤖 Flight Delay Predictor (XGBoost Model)")

    st.markdown(
        "Provide flight details and the trained model will predict the probability "
        "that the flight will be delayed (DELAY_LABEL = 1)."
    )

    # Sidebar selections using original category names
    airlines = sorted(code_maps["AIRLINE_CODE"].keys())
    origins = sorted(code_maps["ORIGIN"].keys())
    dests = sorted(code_maps["DEST"].keys())

    col1, col2 = st.columns(2)

    with col1:
        airline_sel = st.selectbox("Airline Code", airlines)
        origin_sel = st.selectbox("Origin Airport", origins)
        dest_sel = st.selectbox("Destination Airport", dests)
        dep_hour = st.slider("Scheduled Departure Hour (0–23)", 0, 23, 12)

    with col2:
        month = st.slider("Month (1–12)", 1, 12, 7)
        day_of_week = st.slider("Day of Week (1=Mon ... 7=Sun)", 1, 7, 4)
        distance = st.number_input("Distance (miles)", min_value=50, max_value=5000, value=800)
        fl_number = st.number_input("Flight Number", min_value=1, max_value=9999, value=123)

    st.markdown("#### Optional: Expected Schedule / Small Delays")
    col3, col4 = st.columns(2)
    with col3:
        crs_dep_time = st.number_input("CRS Departure Time (HHMM)", min_value=0, max_value=2359, value=900)
        dep_delay = st.number_input("Departure Delay So Far (mins)", min_value=-60, max_value=300, value=0)
    with col4:
        crs_arr_time = st.number_input("CRS Arrival Time (HHMM)", min_value=0, max_value=2359, value=1100)
        # We don't know final arrival delay yet, assume 0 as baseline
        arr_delay = st.number_input("Expected Arrival Delay (mins, can be 0)", min_value=-60, max_value=600, value=0)

    if st.button("Predict Delay Risk"):
        # Build feature row with all model features
        input_dict = {col: 0 for col in feature_cols}

        # Map categorical selections to codes
        input_dict["AIRLINE_CODE"] = code_maps["AIRLINE_CODE"][airline_sel]
        input_dict["ORIGIN"] = code_maps["ORIGIN"][origin_sel]
        input_dict["DEST"] = code_maps["DEST"][dest_sel]

        # Fill numeric features if present in model
        def set_if_exists(name, value):
            if name in input_dict:
                input_dict[name] = value

        set_if_exists("DEP_HOUR", dep_hour)
        set_if_exists("MONTH", month)
        set_if_exists("DAY_OF_WEEK", day_of_week)
        set_if_exists("DISTANCE", distance)
        set_if_exists("FL_NUMBER", fl_number)
        set_if_exists("CRS_DEP_TIME", crs_dep_time)
        set_if_exists("CRS_ARR_TIME", crs_arr_time)
        set_if_exists("DEP_DELAY", dep_delay)
        set_if_exists("ARR_DELAY", arr_delay)

        # Convert to DataFrame in correct column order
        input_df = pd.DataFrame([input_dict])[feature_cols]

        # Predict
        pred_proba = model.predict_proba(input_df)[0, 1]
        pred_class = model.predict(input_df)[0]
        risk_label = label_delay_prob(pred_proba)

        st.markdown("### Prediction Result")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Delay Probability", f"{pred_proba*100:.1f}%")
        col_b.metric("Risk Level", risk_label)
        col_c.metric("Predicted Class", "Delayed" if pred_class == 1 else "On-Time")

        st.info(
            "Note: This is a demo model trained on historical averages; "
            "predictions are not for operational use."
        )
