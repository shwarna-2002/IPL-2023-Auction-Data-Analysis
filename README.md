# IPL 2023 Auction Analysis - Price Prediction

This is my end-to-end Data Science project on IPL 2023 Auction. I have analyzed 568 players which includes sold, unsold and retained players list. As an IPL fan I was curious why some players get 15 Cr and some remain unsold even with good stats, so I tried to find the pattern and predict the price.

### Dataset Used
- File Name: `IPL_Squad_2023_Auction_Dataset.csv`
- Total Rows: 568 players
- Columns: Player Name, Team, Role/TYPE (Batter, Bowler, All-rounder, Wicket-keeper), Age, Base Price, Final Sold Price (COST IN ₹ CR.), Previous IPL matches, Runs, Wickets etc.
- Source: Official IPL 2023 Auction data

### My Complete Project Workflow (Step 1 to 5 in detail)

This is how I implemented everything inside my notebook `IPL-2023-Auction-Data-Analysis.ipynb`. I combined it into one main flow for better understanding:

```python
# ==========================================
# STEP 1: DATA LOADING AND DATA CLEANING
# ==========================================
import pandas as pd
import numpy as np

# Loading the dataset which has 568 players
df = pd.read_csv('IPL_Squad_2023_Auction_Dataset.csv')
print("Original Shape:", df.shape)
print(df.head())

# Initial checks
print(df.info()) # to see data types and null values
print(df.isnull().sum()) # checking how many null values in each column

# Cleaning Part:
# The COST column had values like '₹ 6 Cr' as string, so I had to clean it
df['COST IN ₹ (CR.)'] = df['COST IN ₹ (CR.)'].astype(str).str.replace('₹','').str.replace('Cr','').str.replace('cr','').str.strip()
df['COST IN ₹ (CR.)'] = pd.to_numeric(df['COST IN ₹ (CR.)'], errors='coerce')

# Some players were unsold so their cost was NaN, I filled them with 0 or dropped based on analysis
df['COST IN ₹ (CR.)'].fillna(0, inplace=True)

# Removed duplicates if any
df.drop_duplicates(inplace=True)
print("After Cleaning Shape:", df.shape) # Final cleaned data ready for EDA

# ==========================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

# Q1. Which player role gets more money?
plt.figure(figsize=(8,5))
sns.boxplot(x='TYPE', y='COST IN ₹ (CR.)', data=df)
plt.title('Role vs Final Price')
plt.show()
# Finding: All-rounders median price is highest, then Batters, then Bowlers. Teams pay more for all-rounders because they give both batting and bowling.

# Q2. Does Age affect Price?
sns.scatterplot(x='Age', y='COST IN ₹ (CR.)', data=df, hue='TYPE')
plt.title('Age vs Price')
plt.show()
# Finding: Price increases from age 20 to 31, peak around 28-32, after 34 it starts decreasing. Teams prefer young + experienced, not too old.

# Q3. Base Price vs Final Price relation?
sns.scatterplot(x='Base Price', y='COST IN ₹ (CR.)', data=df)
# Finding: Players who kept higher base price generally have confidence and past performance, and they do get higher final price.

# Q4. Top 10 Costliest Players?
df.sort_values(by='COST IN ₹ (CR.)', ascending=False).head(10)[['Player Name','TYPE','COST IN ₹ (CR.)']]

# Q5. Team-wise spending?
df.groupby('Team')['COST IN ₹ (CR.)'].sum().sort_values(ascending=False).plot(kind='bar')

# ==========================================
# STEP 3: FEATURE ENGINEERING
# ==========================================
from sklearn.preprocessing import LabelEncoder

# Machine learning models need numbers, not text. So I converted categorical to numerical.
le_type = LabelEncoder()
le_team = LabelEncoder()

df['TYPE_ENCODED'] = le_type.fit_transform(df['TYPE']) # Batter=0, Bowler=1 etc.
df['Team_ENCODED'] = le_team.fit_transform(df['Team'])

# Creating new useful features
# Example: Is_Allrounder feature (1 if all-rounder else 0) because from EDA we saw all-rounders get premium
df['Is_Allrounder'] = df['TYPE'].apply(lambda x: 1 if 'All' in x else 0)

# Final feature selection for model - I selected only important numeric features
# I avoided Player Name because it's not useful for prediction
X = df[['Age', 'Base Price', 'TYPE_ENCODED', 'Team_ENCODED', 'Is_Allrounder']] 
y = df['COST IN ₹ (CR.)'] # This is what we want to predict - Final Price

print("Features used:", X.columns.tolist())

# ==========================================
# STEP 4: MODEL BUILDING
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Splitting data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression - Simple model to start with
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("Linear Regression tried")

# Model 2: Random Forest Regressor - This is advanced, works better for non-linear data like auction
# It makes many decision trees and averages them
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest trained with 100 trees")

# ==========================================
# STEP 5: MODEL EVALUATION AND RESULTS
# ==========================================
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Evaluating Linear Regression
lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
print(f"Linear Regression RMSE: {lr_rmse}")

# Evaluating Random Forest - This gave me better result
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"Random Forest RMSE: {rf_rmse}") # My final RMSE was 3.978
print(f"R2 Score: {rf_r2}")
print(f"MAE: {rf_mae}")
# RMSE means on average my prediction is off by 3.978 Cr - which is decent for IPL range 0.2 to 18 Cr

# Finding Top Important Features
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_imp = feature_imp.sort_values(ascending=False)
print("Top Factors affecting price:")
print(feature_imp)

plt.figure(figsize=(8,4))
feature_imp.plot(kind='bar')
plt.title('Top 10 Important Features for Price Prediction')
plt.show()
# Result: Base Price and Is_Allrounder and Age were top 3 most important factors

## How to Run
1. Clone repo: git clone https://github.com/shwarna-2002/IPL-2023-Auction-Data-Analysis.git
2. Install: pip install pandas scikit-learn matplotlib seaborn
3. Open notebook: IPL-2023-Auction-Data-Analysis.ipynb

Author: Shwarna-2002
