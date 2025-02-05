# Housing Price Prediction - Machine Learning Models

## Overview
This project utilizes various **machine learning models** to predict house prices based on multiple features extracted from housing data. The models used include:

- **Logistic Regression**
- **Decision Tree Classifier & Regressor**
- **Random Forest Regressor & Classifier**

The dataset used for training and testing is `train.csv`, which contains housing features and the target variable `SalePrice`.

## Features
The following features are selected for training the models:
- Property characteristics (`MSSubClass`, `LotArea`, `OverallQual`, `OverallCond`, `YearBuilt`, `GrLivArea`, `BedroomAbvGr`, `PoolArea`, `YrSold`)
- Zoning (`MSZoning_RL`, `MSZoning_FV`, `MSZoning_RH`, `MSZoning_RM`)
- Utilities (`Utilities_AllPub`, `Utilities_NoSeWa`)
- Exterior conditions (`ExterQual_Ex`, `ExterQual_Gd`, `ExterQual_TA`, `ExterQual_Fa`, `ExterCond_Fa`, `ExterCond_Po`, `ExterCond_Ex`, `ExterCond_Gd`, `ExterCond_TA`)
- Basement conditions (`BsmtCond_Gd`, `BsmtCond_TA`, `BsmtCond_Fa`, `BsmtCond_Po`)
- Electrical configurations (`Electrical_SBrkr`, `Electrical_FuseA`, `Electrical_FuseF`, `Electrical_FuseP`, `Electrical_Mix`)

The dataset is **preprocessed** using one-hot encoding for categorical features and missing values are dropped.

## Installation
### **Requirements**
Ensure you have **Python 3.x** installed along with the required libraries:
```sh
pip install pandas scikit-learn
```

### **Running the Model**
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/housing-price-prediction.git
   cd housing-price-prediction
   ```
2. Place the dataset `train.csv` in the project directory.
3. Run the script:
   ```sh
   python model.py
   ```
## Code Breakdown
### **1. Data Preprocessing**
```python
import pandas as pd
housing = pd.read_csv("train.csv")
features = ["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea", "BedroomAbvGr", "PoolArea", "YrSold"]
housing = pd.get_dummies(housing)
housing = housing.dropna(axis=0, how='any')
y = housing.SalePrice
X = housing[features]
```

### **2. Train-Test Split**
```python
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
```
### **3. Training and Evaluating Models**
#### **Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
LR = LogisticRegression(random_state=1, max_iter=5000)
LR.fit(train_X, train_y)
data_y = LR.predict(test_X)
print(mean_absolute_error(data_y, test_y))
```

#### **Decision Tree Classifier & Regressor**
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
DTC = DecisionTreeClassifier(max_depth=38, random_state=1)
DTC.fit(train_X, train_y)
data_y_dtc = DTC.predict(test_X)
print(mean_absolute_error(data_y_dtc, test_y))

DTR = DecisionTreeRegressor(max_depth=38, random_state=1)
DTR.fit(train_X, train_y)
data_y_dtr = DTR.predict(test_X)
print(mean_absolute_error(data_y_dtr, test_y))
```

#### **Random Forest Regressor & Classifier**
```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
RFR = RandomForestRegressor(random_state=1)
RFR.fit(train_X, train_y)
data_y_rfr = RFR.predict(test_X)
print(mean_absolute_error(data_y_rfr, test_y))

RFC = RandomForestClassifier(random_state=1, max_depth=15)
RFC.fit(train_X, train_y)
data_y_rfc = RFC.predict(test_X)
print(mean_absolute_error(data_y_rfc, test_y))
```
