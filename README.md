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
