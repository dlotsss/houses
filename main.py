import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

#Housing data
df = pd.read_csv("train.csv")

print("Data frame shape:", df.shape)
print("\nData types and non-null counts:\n", df.info())
print("\nBasic stats:\n", df.describe())

cols_with_many_nans = ["Alley", "PoolQC", "Fence", "MiscFeature"]
df.drop(columns=cols_with_many_nans, inplace=True)

num_features = df.select_dtypes(include=[np.number]).columns
num_imputer = SimpleImputer(strategy="median")
df[num_features] = num_imputer.fit_transform(df[num_features])
