# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    df = df.drop(columns=['customerID'], errors='ignore')
    return df

def encode_categoricals(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df
