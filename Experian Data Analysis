import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
import warnings
import matplotlib.pyplot as plt 

class_weight='balanced'

warnings.filterwarnings("ignore")  # to keep output clean

# Load data files (make sure filenames exactly match!)
account_holder_df = pd.read_csv(r'C:\Users\onuud\Downloads\experian\AcountHolderData.csv')
account_df = pd.read_csv(r'C:\Users\onuud\Downloads\experian\AccountData.csv')
mule_flag_df = pd.read_csv(r'C:\Users\onuud\Downloads\experian\MuleFlag.csv')

# Merge on 'Identifier'
data = account_holder_df.merge(account_df, on='Identifier').merge(mule_flag_df, on='Identifier')

# Convert DateOfBirth to datetime
data['DateOfBirth'] = pd.to_datetime(data['DateOfBirth'], dayfirst=True)

# Calculate age assuming data snapshot date (e.g., today or a fixed date)
snapshot_date = pd.to_datetime('2025-01-01')  # or use pd.Timestamp.today()

data['Age'] = (snapshot_date - data['DateOfBirth']).dt.days // 365

# Define age bins and labels
age_bins = [0, 17, 24, 35, 45, 60, 100]
age_labels = ['0-17', '18-24', '25-35', '36-45', '46-60', '60+']
data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

# Define income bins and labels
income_bins = [0, 10000, 20000, 30000, 40000, 60000, 80000, 100000]
income_labels = ['0-10k', '10k-20k', '20k-30k', '30k-40k', '40k-60k', '60k-80k', '80k+']
data['IncomeGroup'] = pd.cut(data['Income'], bins=income_bins, labels=income_labels, right=False)

age_group_summary = data.groupby('AgeGroup')['MuleAccount'].sum()
print("Mule Accounts by Age Group:")
print(age_group_summary)

# Group by Gender and sum MuleAccount
gender_summary = data.groupby('Gender')['MuleAccount'].sum()
print("\nMule Accounts by Gender:")
print(gender_summary)

# Group by combined characteristics
characteristics = ['AgeGroup', 'Gender']
mule_summary = data.groupby(characteristics)['MuleAccount'].sum().reset_index()
mule_summary = mule_summary.sort_values(by='MuleAccount', ascending=False)
print("\nTop 5 Characteristics with highest Mule Accounts:")
print(mule_summary.head(5))

# Drop original DateOfBirth column
data = data.drop(columns=['DateOfBirth'])

# Show basic info
print("Data shape:", data.shape)
print("Sample data:")
print(data.head())

# Prepare features (X) and target (y)
# Drop Identifier and MuleFlag from features
X = data.drop(columns=['Identifier', 'MuleAccount'])

# Handle categorical columns - simple encoding
# Find categorical columns by dtype
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols)

y = data['MuleAccount']

# Remove rows with missing target values
data = data.dropna(subset=['MuleAccount'])

# Now redefine X and y after dropping
X = data.drop(columns=['Identifier', 'MuleAccount'])
y = data['MuleAccount']

X = data.drop(columns=['Identifier', 'MuleAccount'])

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)  # drop_first=True avoids dummy variable trap

y = data['MuleAccount']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 10 Predictive Features:")
print(importances.head(10))

# Plot bar chart for AgeGroup vs MuleAccounts
plt.figure(figsize=(10, 5))
age_group_summary.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Mule Accounts by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Mule Accounts')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot bar chart for Gender vs MuleAccounts
plt.figure(figsize=(6, 4))
gender_summary.plot(kind='bar', color='purple', edgecolor='black')
plt.title('Number of Mule Accounts by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Mule Accounts')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
