# Mule Account Detection Pipeline

This project implements a machine learning pipeline to analyze and classify mule accounts based on demographic and account data. It includes data preprocessing, exploratory analysis, feature engineering, model training, and evaluation using a Random Forest classifier.

## Project Overview

The pipeline:

- Loads and merges data from multiple sources
- Cleans and preprocesses data
- Engineers features such as age groups and income brackets
- Performs exploratory analysis (grouped summaries and visualizations)
- Trains a Random Forest classifier
- Evaluates the model's performance
- Highlights the most important predictive features

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- matplotlib

## Data Inputs

- `AcountHolderData.csv`
- `AccountData.csv`
- `MuleFlag.csv`

Ensure these CSV files are available and correctly named.

## Main Steps

### 1. Data Loading & Merging
- Data from account holders, account info, and mule flags is merged on `Identifier`.

### 2. Preprocessing
- Converts `DateOfBirth` to datetime
- Calculates age and bins into age groups
- Bins income into income brackets
- Removes `DateOfBirth` after use

### 3. Exploratory Analysis
- Displays mule account counts by age group and gender
- Prints top combinations of features associated with mule accounts

### 4. Feature Engineering
- One-hot encodes categorical features
- Removes unnecessary or non-numeric columns

### 5. Modeling
- Splits data into training and test sets
- Trains a Random Forest classifier
- Evaluates performance with a classification report

### 6. Results
- Prints top 10 predictive features based on importance
- Shows bar plots of mule account distribution by age group and gender

## Output

- Console summary of data and model performance
- Visualizations for age and gender analysis
- Feature importance rankings

## Example Visualizations

- Mule accounts by age group
- Mule accounts by gender
