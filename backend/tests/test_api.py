# backend/tests/test_api.py
from fastapi.testclient import TestClient
from backend.app import app # Import the FastAPI app instance

# Create a test client for your FastAPI application
client = TestClient(app)

def test_health_check():
    """
    Test the /health endpoint to ensure the API is running.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    # The model_loaded status might be False if the model file hasn't been generated yet
    # For a robust test, you might want to mock the MLService or ensure the model exists.
    # For now, we'll just check if the key exists.
    assert "model_loaded" in response.json()

def test_predict_fraud_safe_transaction():
    """
    Test the /predict_fraud endpoint with a transaction expected to be safe.
    """
    # Example of a transaction data that should be classified as 'Not Fraud'
    safe_transaction_data = {
        "amount": 100.00,
        "transaction_frequency_24h": 1,
        "location_risk_score": 0.5,
        "time_of_day_hour": 14,
        "is_international": 0,
        "ip_country_mismatch": 0
    }
    response = client.post("/predict_fraud", json=safe_transaction_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert data["is_fraud"] == False # Expecting not fraud for this data
    assert data["fraud_probability"] < 0.5 # Expecting low probability of fraud

def test_predict_fraud_potentially_fraudulent_transaction():
    """
    Test the /predict_fraud endpoint with a transaction expected to be fraudulent.
    """
    # Example of a transaction data that should be classified as 'Fraud'
    fraudulent_transaction_data = {
        "amount": 1500.00,
        "transaction_frequency_24h": 8,
        "location_risk_score": 9.0,
        "time_of_day_hour": 2,
        "is_international": 1,
        "ip_country_mismatch": 1
    }
    response = client.post("/predict_fraud", json=fraudulent_transaction_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert data["is_fraud"] == True # Expecting fraud for this data
    assert data["fraud_probability"] > 0.5 # Expecting high probability of fraud

def test_predict_fraud_invalid_input():
    """
    Test the /predict_fraud endpoint with invalid input data.
    """
    # Missing required fields
    invalid_transaction_data = {
        "amount": "not_a_number", # Invalid type
        "transaction_frequency_24h": 1
    }
    response = client.post("/predict_fraud", json=invalid_transaction_data)
    assert response.status_code == 422 # Unprocessable Entity (Pydantic validation error)

    # Missing a required field
    missing_field_data = {
        "amount": 100.00,
        "transaction_frequency_24h": 1,
        "location_risk_score": 0.5,
        "time_of_day_hour": 14,
        "is_international": 0
        # ip_country_mismatch is missing
    }
    response = client.post("/predict_fraud", json=missing_field_data)
    assert response.status_code == 422 # Unprocessable Entity
