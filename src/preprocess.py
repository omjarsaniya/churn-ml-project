import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df.drop("customerID", axis=1, inplace=True)
    
    return df

def encode_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def split_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return X, y