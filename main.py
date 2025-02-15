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

cat_features = df.select_dtypes(include=["object"]).columns
for cat_col in cat_features:
    df[cat_col] = df[cat_col].fillna("None")

df = df[df["SalePrice"] < 700000]

corr_matrix = df.corr()
top_corr = corr_matrix["SalePrice"].abs().sort_values(ascending=False).head(15)
print("\nTop correlated features with SalePrice:\n", top_corr)

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "YearBuilt",
    "YearRemodAdd",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd"
]

X = df[features]
y = df["SalePrice"]

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)
preds = model.predict(test_X)

mae = mean_absolute_error(test_y, preds)

print("\nRandomForestRegressor MAE:", mae)

scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
print("Cross-validated MAE:", -1 * np.mean(scores))
