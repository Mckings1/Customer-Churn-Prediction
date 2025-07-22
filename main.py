import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

def main():
    # Load your dataset
    df = pd.read_csv("C:\Users\Envy\Documents\WEBDEV\Data Science-Analytics\customer-churn-prediction\data\telco_customer_churn.csv")
    print("Data loaded successfully:", df.shape)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessed.")

    # Train model
    model = train_model(X_train, y_train)
    print("Model training complete.")

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
