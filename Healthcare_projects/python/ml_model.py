import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("../output", exist_ok=True)

#---------------------------------------
# 1. Loading data
#---------------------------------------
print("Loading the data from SQLite...")
conn = sqlite3.connect("../sql/sleep_health.db")
df = pd.read_sql_query("SELECT * FROM tbl_sleep", conn)
conn.close()

#---------------------------------------
# 2. Target variable (What do we wanr to predict?)
#---------------------------------------
y = df["felt_rested"]

#---------------------------------------
# 3. Features (What data is the modell allowed to use?)
#---------------------------------------
# Removing 'person_id' (useless) and 'cognituve_performance_score' (too direct a correlation).
# Also removing 'sleep_disorder_risk', since the goal is to predict behavior.
X = df.drop(columns=["person_id", "felt_rested", "cognitive_performance_score", "sleep_disorder_risk"])

# Converting categorical variables (text) into numbers (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

#---------------------------------------
# 4. Split training and test data (80% training, 20% testing)
#---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#--------------------------------------
# 5. Training the model (Random Forest)
#---------------------------------------
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

#---------------------------------------
# 6. Prediction on the test data
#---------------------------------------
y_pred = model.predict(X_test)

#---------------------------------------
# 7. Output accuarcy
#---------------------------------------
accuarcy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuarcy * 100:.2f}%")
print("\nDatailed report:")
print(classification_report(y_test, y_pred))

#---------------------------------------
# 8. Which factors are the most important? (Feature Importance)
#---------------------------------------
importance = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)

# Visualization of the Top 10 Features
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, hue="Feature", palette="viridis", legend=False)
plt.title("Top 10 influence factors for restful sleep")
plt.xlabel("Importance")
plt.ylabel("Factor")
plt.tight_layout()
plt.savefig("../output/ml_feature_importance.png")
print("\nDiagram of the most important factors saved: output/ml_feature_importance.png")