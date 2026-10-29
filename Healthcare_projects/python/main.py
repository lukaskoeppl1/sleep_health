import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3
import os

#---------------------------------------
# 1. Loading and overview the data
#---------------------------------------

sleep_data = pd.read_csv("../data/sleep_health_dataset.csv")
print(sleep_data.head())
print(sleep_data.describe())
print(sleep_data.info())
print("\nNumber of missing values: ", sleep_data.isnull().sum())
print("\nAmount of duplicates in the sleep dataset: ", sleep_data.duplicated().sum())

#---------------------------------------
# 2. Cleaning the data
#---------------------------------------
sleep_data_cleaned = sleep_data.drop_duplicates()

print(f"\nLines before cleaning: {len(sleep_data)}")
print(f"\nLines before cleaning: {len(sleep_data_cleaned)}")

# Secure check: Shows the first 2 lines of the cleaned table
print("\nSecurity check before saving (should be real data):")
print(sleep_data_cleaned.head(2))

#---------------------------------------
# 3. SQLite (saving the data in SQLite)
#---------------------------------------
print("Loading the data into SQLite")
db_path = "../sql/sleep_health.db"
conn = sqlite3.connect(db_path)

# Loading the data into "tbl_sleep" table.
sleep_data_cleaned.to_sql("tbl_sleep", conn, if_exists="replace", index=False)
conn.close()

print(f"Data successfully loaded into SQLite! ({db_path})")