# backend/ml_service.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
import os

# Define the path where the model will be saved/loaded
# This path is relative to the project root, assuming ml_service.py is in backend/
# and the model is in data/models/
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models', 'fraud_detection_model.joblib')

class MLService:
    def __init__(self, model_path: str = MODEL_SAVE_PATH):
        """
        Initializes the MLService by loading the pre-trained model.
        If the model does not exist, it will train and save a new one (for demonstration).
        """
        self.model_path = model_path
        self.model = None
        self._load_or_train_model()

    def _generate_synthetic_data(self, num_samples=10000):
        """
        Generates synthetic data for demonstration purposes.
        In a real scenario, this would be replaced by loading actual data.
        """
        print("Generating synthetic data for demonstration...")
        np.random.seed(42) # for reproducibility

        data = {
            'amount': np.random.normal(500, 200, num_samples),
            'transaction_frequency_24h': np.random.poisson(3, num_samples),
            'location_risk_score': np.random.rand(num_samples) * 10,
            'time_of_day_hour': np.random.randint(0, 24, num_samples),
            'is_international': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
            'ip_country_mismatch': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
            'fraud': np.zeros(num_samples, dtype=int)
        }
        df = pd.DataFrame(data)

        num_fraud = int(num_samples * 0.02)
        fraud_indices = np.random.choice(df.index, num_fraud, replace=False)
        df.loc[fraud_indices, 'fraud'] = 1
        df.loc[fraud_indices, 'amount'] = np.random.normal(1500, 500, num_fraud)
        df.loc[fraud_indices, 'transaction_frequency_24h'] = np.random.poisson(8, num_fraud)
        df.loc[fraud_indices, 'location_risk_score'] = np.random.rand(num_fraud) * 5 + 5
        df.loc[fraud_indices, 'is_international'] = np.random.choice([0, 1], num_fraud, p=[0.3, 0.7])
        df.loc[fraud_indices, 'ip_country_mismatch'] = np.random.choice([0, 1], num_fraud, p=[0.2, 0.8])
        
        return df

    def _train_model(self, df):
        """
        Trains the RandomForestClassifier model.
        """
        print("Training the RandomForestClassifier model...")
        X = df.drop('fraud', axis=1)
        y = df['fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        print("Model training complete. Evaluation on test set:")
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
        
        return model

    def _load_or_train_model(self):
        """
        Attempts to load the model. If not found, trains a new model and saves it.
        """
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            try:
                self.model = joblib.load(self.model_path)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}. Retraining model.")
                df = self._generate_synthetic_data()
                self.model = self._train_model(df)
                joblib.dump(self.model, self.model_path)
                print(f"New model trained and saved to {self.model_path}")
        else:
            print(f"Model not found at {self.model_path}. Training a new model...")
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            df = self._generate_synthetic_data()
            self.model = self._train_model(df)
            joblib.dump(self.model, self.model_path)
            print(f"New model trained and saved to {self.model_path}")

    def predict(self, transaction_data: dict):
        """
        Makes a fraud prediction for a single transaction.
        Args:
            transaction_data (dict): A dictionary containing transaction features.
                                     Keys must match the features used in training.
        Returns:
            tuple: (prediction_label, fraud_probability)
        """
        if self.model is None:
            raise RuntimeError("ML model is not loaded.")

        # Convert the incoming dictionary to a DataFrame for prediction
        # Ensure the order of columns matches the training data
        # The order of columns from synthetic data generation is:
        # 'amount', 'transaction_frequency_24h', 'location_risk_score',
        # 'time_of_day_hour', 'is_international', 'ip_country_mismatch'
        
        # Create a DataFrame from the single transaction data
        # Ensure the columns are in the correct order as expected by the model
        features = ['amount', 'transaction_frequency_24h', 'location_risk_score',
                    'time_of_day_hour', 'is_international', 'ip_country_mismatch']
        
        # Create a DataFrame with a single row
        input_df = pd.DataFrame([transaction_data], columns=features)

        prediction_label = self.model.predict(input_df)[0]
        fraud_probability = self.model.predict_proba(input_df)[:, 1][0] # Probability of being the positive class (fraud)

        return prediction_label, fraud_probability

# Example of how to run this script to train and save the model
if __name__ == "__main__":
    print("Running ml_service.py directly to train and save the model...")
    service = MLService()
    print("\nMLService initialized. Model is ready.")

    # Example of using the predict method
    print("\nTesting prediction with example data:")
    # Safe transaction
    safe_transaction = {
        'amount': 75.50,
        'transaction_frequency_24h': 1,
        'location_risk_score': 1.2,
        'time_of_day_hour': 10,
        'is_international': 0,
        'ip_country_mismatch': 0
    }
    pred_safe, prob_safe = service.predict(safe_transaction)
    print(f"Safe transaction: Prediction={'Fraud' if pred_safe else 'Not Fraud'}, Probability={prob_safe:.4f}")

    # Potentially fraudulent transaction
    fraudulent_transaction = {
        'amount': 1800.00,
        'transaction_frequency_24h': 7,
        'location_risk_score': 9.5,
        'time_of_day_hour': 3,
        'is_international': 1,
        'ip_country_mismatch': 1
    }
    pred_fraud, prob_fraud = service.predict(fraudulent_transaction)
    print(f"Fraudulent transaction: Prediction={'Fraud' if pred_fraud else 'Not Fraud'}, Probability={prob_fraud:.4f}")
