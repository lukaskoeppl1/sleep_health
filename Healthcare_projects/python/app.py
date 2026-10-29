import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#---------------------------------------
# 1. Setting the data up
#---------------------------------------
st.set_page_config(page_title="Sleep Health Dashboard", layout="wide")
st.title("Sleep Health & Daily Performance")
st.markdown("Interactive analysis of 100.000 sleeping profiles.")

#---------------------------------------
# 2. Database connection (Path to sql-directory)
#---------------------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect("../sql/sleep_health.db")
    df = pd.read_sql_query("SELECT * FROM tbl_sleep", conn)
    conn.close()
    return df

@st.cache_resource
def train_model(df):
    y = df["felt_rested"]
    X = df.drop(columns=["person_id", "felt_rested", "cognitive_performance_score", "sleep_disorder_risk", "persona", "cluster"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns

df = load_data()
model, feature_columns = train_model(df)

@st.cache_data
def add_personas(df):
    df_copy = df.copy()
    features = ["sleep_duration_hrs", "stress_score", "sleep_quality_score", "caffeine_mg_before_bed", "screen_time_before_bed_mins"]
    X_cluster = df_copy[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_copy["cluster"] = kmeans.fit_predict(X_scaled)

    summary = df_copy.groupby("cluster")[["sleep_duration_hrs", "stress_score"]].mean()
    best_sleep = summary["sleep_duration_hrs"].idxmax()
    worst_stress = summary["stress_score"].idxmax()

    def name_cluster(c):
        if c == best_sleep: return "Well-Rested"
        elif c == worst_stress: return "Highly Stressed"
        else: return "Average Sleeper"

    df_copy["persona"] = df_copy["cluster"].apply(name_cluster)
    return df_copy

df = add_personas(df)

# Databaseconnection for the SQL-Querys in Tab 1
conn = sqlite3.connect("../sql/sleep_health.db")

#---------------------------------------
# 3. Sidebar Filter (Applies for both tabs)
#---------------------------------------
st.sidebar.header("Filter options")
jobs = pd.read_sql_query("SELECT DISTINCT occupation FROM tbl_sleep", conn)["occupation"].tolist()
conn.close()

choose_job = st.sidebar.multiselect("Choose job", jobs, default=jobs[:5])

personas = df["persona"].unique().tolist()
choose_persona = st.sidebar.multiselect("Choose Persona", personas, default=personas)


#---------------------------------------
# 4. Creating Tabs
#---------------------------------------
tab1, tab2 = st.tabs(["Data Analysis", "ML Prediction"])

#---------------------------------------
# TAB 1: Data Analysis
#---------------------------------------
with tab1:
    if choose_job and choose_persona:
        df_filtered = df[(df["occupation"].isin(choose_job)) & (df["persona"].isin(choose_persona))]
    else:
        df_filtered = df

    if not df_filtered.empty:
        avg_sleep = df_filtered['sleep_duration_hrs'].mean()
        avg_quality = df_filtered['sleep_quality_score'].mean()
        avg_stress = df_filtered['stress_score'].mean()

        total_people = len(df_filtered)
        healthy_people = len(df_filtered[df_filtered['sleep_disorder_risk'] == "Healthy"])
        healthy_rate = (healthy_people / total_people) * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ø sleep duration",f"{avg_sleep:.1f} Std.")
        k2.metric("Ø sleep quality",f"{avg_quality:.1f} / 10")
        k3.metric("Ø stress level",f"{avg_stress:.1f} / 10")
        k4.metric("healthy rate",f"{healthy_rate:.1f}%")

        st.markdown("---")

        col1, col2 = st.columns(2)
                           
        with col1:
            st.subheader("Average sleep duration per job")
            df_bar = df_filtered.groupby("occupation")["sleep_duration_hrs"].mean().sort_values(ascending=True).reset_index()
            st.bar_chart(df_bar.set_index("occupation"), horizontal=True, color="#2E86C1")
                           
        with col2:
            st.subheader("Stress vs. cognitive performance")
            df_scatter = df_filtered.sample(min(5000, len(df_filtered)))
            st.scatter_chart(
                data=df_scatter,
                x="stress_score",
                y="cognitive_performance_score",
                color="occupation",
                use_container_width=True
            )
                  
        st.markdown("---")
        st.subheader("Correlation Heatmap")
        st.markdown("This heatmap shows how the most important factors influence each other.")
                  
        heatmap_cols = ["sleep_duration_hrs", "sleep_quality_score", "stress_score", "caffeine_mg_before_bed", "alcohol_units_before_bed", "screen_time_before_bed_mins", "cognitive_performance_score"]
                  
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_filtered[heatmap_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

#---------------------------------------
# TAB 2: Interactive ML-Prediction
#---------------------------------------
with tab2:
    st.subheader("Will this person wake up rested?")
    st.markdown("Adjust the lifestyle factors below to see how the Machine Learning model predicts their sleep quality.")

    # Slider in 3 columns
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        input_sleep_dur = st.slider("Sleep Duration (hrs)", 3.0, 10.5, 7.0, step=0.1)
        input_sleep_qual = st.slider("Sleep Quality (1-10)", 1.0, 10.0, 7.0, step=0.1)
        input_stress = st.slider("Stress Score (0-10)", 0.0, 10.0, 4.0, step=0.1)
        input_work = st.slider("Work Hours", 0.0, 16.0, 8.0, step=0.5)

    with col_b:
        input_caffeine = st.slider("Caffeine (mg)", 0, 500, 100, step=10)
        input_alcohol = st.slider("Alcohol (units)", 0.0, 5.0, 0.0, step=0.5)
        input_screen = st.slider("Screen time before bed (mins)", 0, 180, 60, step=5)
        input_rem = st.slider("REM Sleep (%)", 10.0, 30.0, 20.0, step=0.1)

    with col_c:
        input_deep = st.slider("Deep Sleep (%)", 5.0, 25.0, 15.0, step=0.1)
        input_wake = st.slider("Wake Episodes", 0, 10, 2)
        input_bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)
        input_steps = st.slider("Steps that day", 0, 30000, 8000, step=500)

    # Prediction-Button
    if st.button("Predict", type="primary"):
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

        # Enter values from the sliders
        input_data["sleep_duration_hrs"] = input_sleep_dur
        input_data["sleep_quality_score"] = input_sleep_qual
        input_data["stress_score"] = input_stress
        input_data["work_hours_that_day"] = input_work
        input_data["caffeine_mg_before_bed"] = input_caffeine
        input_data["alcohol_units_before_bed"] = input_alcohol
        input_data["screen_time_before_bed_mins"] = input_screen
        input_data["rem_percentage"] = input_rem
        input_data["deep_sleep_percentage"] = input_deep
        input_data["wake_episodes_per_night"] = input_wake
        input_data["bmi"] = input_bmi
        input_data["steps_that_day"] = input_steps

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.markdown("---")
        if prediction == 1:
            st.success(f" **Prediction: This person WILL wake up rested.** (Probability: {probability[1]*100:.1f}%)")
        else:
            st.error(f" **Prediction: This person will NOT wake up rested.** (Probability: {probability[0]*100:.1f}%)")

    st.markdown("---")
    st.subheader("Top 10 factors for restful sleep")
    st.markdown("This chart shows which features the Machine Learning model considers most important to predict if someone wakes up rested.")

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(
        feature_importance_df.set_index("Feature"),
        horizontal=True,
        color="#E74C3C"
    )