# src/model.py
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def predict_proba(model, X_test):
    return model.predict_proba(X_test)[:, 1]
