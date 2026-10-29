SLEEP HEALTH & DAILY PERFORMANCE DASHBOARD

An interactive data science and machine learning web application built with Streamlit. This project analyzes a dataset of 100,000 sleep profiles to uncover insights into how lifestyle, occupation, and stress affect sleep quality and cognitive performance. It also features a Machine Learning model that predicts the probability of waking up rested.

FEATURES

Data Pipeline (ETL): Raw CSV data is cleaned and loaded into a local SQLite database for efficient querying.

Interactive Data Analysis:
Dynamic KPIs that update based on user filters (Occupation, Persona).
Bar charts comparing average sleep duration across different jobs.
Scatter plots analyzing the relationship between stress and cognitive performance.
Correlation Heatmap showing the interconnections between sleep metrics and lifestyle factors.

Unsupervised Learning (K-Means Clustering): Automatically groups the 100,000 individuals into 3 distinct personas: Well-Rested, Highly Stressed, and Average Sleeper.

Supervised Learning (Random Forest): A classification model that predicts whether a person will wake up rested based on adjustable lifestyle sliders (sleep duration, stress, caffeine, screen time, etc.).

Feature Importance: Visualizes the top 10 most influential factors the ML model uses to make its predictions.

POWER BI BUSINESS INTELLIGENCE DASHBOARD

In addition to the Streamlit web application, the same dataset was used to build a classical Business Intelligence dashboard in Microsoft Power BI.

FEATURES OF THE POWER BI REPORT:

KPI Cards:
Displays the overall Average Sleep Duration, Average Stress Score, Average Cognitive Performance, and a dynamic "Healthy Rate" percentage.

DAX Implementation:
Custom DAX measure using CALCULATE and DIVIDE to calculate the percentage of healthy individuals dynamically.

Interactive Bar Chart:
Visualizes the average sleep duration across different occupations (revealing that "Lawyers" sleep the least and "Retired" individuals sleep the most).

Scatter Plot Analysis:
Plots Average Stress Scores against Average Cognitive Performance Scores, clearly showing a negative correlation between high stress and low cognitive performance.

Interactive Slicer:
Allows users to filter the entire dashboard by occupation to explore specific job-related sleep patterns.

File Location:
powerbi/sleep_health_dashboard.pbix (or the saved screenshot powerbi_dashboard.png).

TECH STACK

Language: Python 3

Data Manipulation: Pandas, NumPy

Database: SQLite

Machine Learning: Scikit-Learn (RandomForest, KMeans, StandardScaler)

Visualization: Matplotlib, Seaborn, Streamlit Native Charts, Power BI

Web Framework: Streamlit

DATASET INFORMATION

The data used in this project is the Sleep Health & Daily Performance Dataset (via Kaggle).

Size: 100,000 rows, 32 columns.

Features: Demographics (Age, Gender, Occupation), Sleep Architecture (REM %, Deep Sleep %, Sleep Latency), Lifestyle (Caffeine, Alcohol, Screen Time, Steps), and Psychological factors (Stress Score, Chronotype).

Target Variable: felt_rested (Binary: 0 = No, 1 = Yes).

PROJECT STRUCTURE

text
├── data/
│   └── sleep_health_dataset.csv    # Raw dataset from Kaggle
├── python/
│   ├── main.py                     # ETL script (loads CSV to SQLite)
│   ├── ml_model.py                 # ML training script (optional)
│   └── app.py                      # Streamlit Dashboard Application
├── sql/
│   └── sleep_health.db             # Generated SQLite Database
├── powerbi/
│   └── sleep_health_dashboard.pbix # Power BI Report File
└── README.md

INSTALLATION AND SETUP

1. Clone the repository:
bash
git clone https://github.com/your-username/sleep-health-dashboard.git
cd sleep-health-dashboard

2. Install dependencies:
Make sure you have Python installed. Then, install the required libraries:

bash
pip install pandas numpy matplotlib seaborn streamlit scikit-learn

3. Prepare the Database:
Place the raw sleep_health_dataset.csv inside the data/ folder. Then, run the ETL script to create the SQLite database:

bash
cd python
python main.py

(This will generate sleep_health.db in the sql/ folder).

4. Run the Dashboard:
While still in the python/ folder, execute:

bash
python -m streamlit run app.py

The application will open automatically in your default web browser (usually at http://localhost:8501).

5. Open the Power BI Dashboard:
Open Microsoft Power BI Desktop and load the file located at powerbi/sleep_health_dashboard.pbix.

KEY INSIGHTS FOUND IN THIS PROJECT

The Sleep Gap: The model and analysis confirm a significant gap in sleep duration based on occupation. Lawyers and Nurses tend to sleep the least, while Retired individuals sleep the most.

Model Interpretability: By analyzing the Feature Importance, the Random Forest model confirmed that Sleep Duration and Sleep Quality are by far the most critical factors for feeling rested, followed closely by Stress Levels.

Stress vs. Cognition: The scatter plot reveals a clear negative correlation between high stress scores and low cognitive performance scores, proving that psychological state heavily impacts daily performance.

AUTHOR

Lukas Köppl Markwart
Apprentice IT Specialist for Data and Process Analysis (Fachinformatiker für Daten- und Prozessanalyse).