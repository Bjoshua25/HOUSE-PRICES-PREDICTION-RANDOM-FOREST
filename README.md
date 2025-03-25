# **HOUSE PRICES PREDICTION | RANDOM FOREST**  
*Ensemble Learning for Real Estate Valuation*  

## **INTRODUCTION**  
Predicting house prices is a critical aspect of **real estate analysis**. This project applies **Random Forest Regression**, an ensemble learning technique, to predict house prices based on **lot size**.  

By leveraging **multiple decision trees**, Random Forest provides a **robust and accurate model** for estimating property values.  

---

## **PROBLEM STATEMENT**  
House prices fluctuate due to multiple factors, making prediction complex.  
This project aims to:  
- **Analyze historical house price data** to identify patterns.  
- **Train a Random Forest model** to predict house prices based on lot size.  
- **Evaluate model performance** using appropriate regression metrics.  

---

## **SKILL DEMONSTRATION**  
- **Data Preprocessing & Feature Engineering**  
- **Exploratory Data Analysis (EDA) & Visualization**  
- **Random Forest Regression Modeling**  
- **Hyperparameter Tuning & Model Optimization**  
- **Model Evaluation (MSE, RMSE, R² Score)**  

---

## **DATA SOURCING**  
The dataset is sourced from [Explore-AI Public Data](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/house_price_by_area.csv) and includes:  

### **1. House Features**  
- **Lot Area (sq. meters)** – Size of the property.  
- **Sale Price ($)** – Final selling price of the house.  

---

## **EXPLORATORY DATA ANALYSIS (EDA)**  
EDA was performed to **understand the relationship between lot area and house price**.  

### **1. Data Overview**  
- **Checked dataset structure** using `.info()` and `.describe()`.  
- **Handled missing values** and outliers.  

### **2. Price Trends Analysis**  
- **Scatter Plot**:  
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['LotArea'], y=df['SalePrice'])
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.title('House Prices vs Lot Area')
plt.show()
```
- **Key Insight:** Larger lot areas generally correspond to higher house prices.  

---

## **RANDOM FOREST MODEL**  
A **Random Forest Regression Model** was trained to predict house prices.  

### **1. Model Implementation**  
- **Independent Variable (`X`)**: Lot Area  
- **Dependent Variable (`y`)**: Sale Price  
- **Model Used**: `sklearn.ensemble.RandomForestRegressor`  

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### **2. Model Evaluation**  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score (Explained Variance)**  

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## **MODEL INTERPRETATION & VISUALIZATION**  
### **1. Feature Importance**  
Identifying the most **impactful factors on house price predictions**:  
```python
import pandas as pd

feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.show()
```
- **Key Finding:** Lot area is the primary determinant of house price in this dataset.  

---

## **CONCLUSION**  
1. **Lot Area significantly influences house prices**.  
2. **Random Forest Regression effectively predicts property values** with high accuracy.  
3. Future improvements should include **additional features (location, amenities, neighborhood factors)** for better predictions.  

---

## **HOW TO RUN THE PROJECT**  
### **1. Prerequisites**  
Ensure you have Python installed along with required libraries:  
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```
### **2. Clone the Repository**  
```bash
git clone https://github.com/yourusername/House-Prices-Prediction-Random-Forest.git
cd House-Prices-Prediction-Random-Forest
```
### **3. Run the Jupyter Notebook**  
```bash
jupyter notebook Introduction_to_random_forests_examples.ipynb
```
