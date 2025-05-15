import pandas as pd
import pickle
import xgboost as xgb
import os
import numpy as np
from pgmpy.inference import VariableElimination

def predict_credit_score_xgb(input_data):
    """
    Load the trained XGBoost model and label encoders to predict the credit score for new data.
    
    Args:
        input_data (dict): Dictionary containing feature values for prediction
        
    Returns:
        dict: Input data with the predicted credit score added
    """
    # Define paths
    base_path = "/app/models"
    model_path = os.path.join(base_path, "xgb_model.json")
    credit_score_encoder_path = os.path.join(base_path, "Credit_Score_label_encoder.pkl")
    
    # Load the trained model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Load the Credit_Score label encoder
    with open(credit_score_encoder_path, "rb") as f:
        credit_score_encoder = pickle.load(f)
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction_encoded = model.predict(input_df)[0]
    
    # # Convert prediction back to original label
    # prediction_decoded = credit_score_encoder.inverse_transform([prediction_encoded])[0]

    # Convert NumPy type to standard Python type
    if isinstance(prediction_encoded, np.integer):
        prediction_encoded = int(prediction_encoded)
    elif isinstance(prediction_encoded, np.floating):
        prediction_encoded = float(prediction_encoded)
    elif isinstance(prediction_encoded, np.ndarray):
        prediction_encoded = prediction_encoded.item()
    elif isinstance(prediction_encoded, np.bool_):
        prediction_encoded = bool(prediction_encoded)
    elif isinstance(prediction_encoded, str):
        pass 
    
    # Add prediction to the input data
    result = input_data.copy()
    result["Credit_Score"] = prediction_encoded
    
    return result


def predict_credit_score_bayesian(input_data):
    """
    Load the trained Bayesian Network model and discretizers to predict the credit score for new data.
    
    Args:
        input_data (dict): Dictionary containing feature values for prediction
        
    Returns:
        dict: Input data with the predicted credit score added
    """
    # Define paths
    base_path = "/app/models"
    model_path = os.path.join(base_path, "bayes_model.pkl")
    age_discretizer_path = os.path.join(base_path, "age_discretizer.pkl")
    income_discretizer_path = os.path.join(base_path, "income_discretizer.pkl")
    
    # Load the trained model and discretizers
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open(age_discretizer_path, "rb") as f:
        age_disc = pickle.load(f)
    
    with open(income_discretizer_path, "rb") as f:
        income_disc = pickle.load(f)
    
    # Create inference engine
    infer = VariableElimination(model)
    
    # Process input data for Bayesian model
    processed_input = input_data.copy()
    
    # Discretize Age
    if 'Age' in processed_input:
        age_val = pd.DataFrame([[processed_input['Age']]], columns=['Age'])
        processed_input['Age'] = int(age_disc.transform(age_val)[0][0])
    
    # Discretize Annual_Income
    if 'Annual_Income' in processed_input:
        inc_val = pd.DataFrame([[processed_input['Annual_Income']]], columns=['Annual_Income'])
        processed_input['Annual_Income'] = int(income_disc.transform(inc_val)[0][0])
    
    # Filter evidence to match the model's nodes
    valid_nodes = set(model.nodes())
    filtered_input = {k: v for k, v in processed_input.items() if k in valid_nodes}
    
    # Perform inference
    result = infer.query(variables=["Credit_Score"], evidence=filtered_input)
    
    # Get the highest probability class (0, 1, or 2)
    probabilities = [result.values[i] for i in range(3)]
    prediction = int(np.argmax(probabilities))
    
    # Add prediction to the input data
    result_dict = input_data.copy()
    result_dict["Credit_Score_Bayesian"] = prediction
    
    return result_dict


def predict_credit_score_combined(data):
    # Call the first prediction model
    result1 = predict_credit_score_xgb(data)
    
    # Call the Bayesian prediction model
    result2 = predict_credit_score_bayesian(data)
    
    # Extract credit scores from both results
    score1 = result1['Credit_Score']
    score2 = result2['Credit_Score_Bayesian']
    
    # Calculate mean and floor down to integer
    mean_score = int((score1 + score2) / 2)
    
    # Create combined result (copy of input data with added score)
    combined_result = data.copy()
    combined_result['Credit_Score'] = mean_score
    
    return combined_result



# Example usage
if __name__ == "__main__":
    # Sample input data
    new_data = {
        "Age": 50,
        "Occupation": 15,
        "Annual_Income": 100000.0,
        "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 1,
        "Interest_Rate": 1,
        "Num_of_Loan": 9,
        "Num_of_Delayed_Payment": 0,
        "Auto Loan": 0,
        "Credit-Builder Loan": 0,
        "Debt Consolidation Loan": 0,
        "Home Equity Loan": 0,
        "Mortgage Loan": 0,
        "Not Specified": 0,
        "Payday Loan": 0,
        "Personal Loan": 0,
        "Student Loan": 0
    }
    
    # # Get prediction
    # result = predict_credit_score_xgb(new_data)
    # print(f"Predicted Credit Score: {result['Credit_Score']}")
    # print("Complete Result:")
    # print(result)

    # bayes_result = predict_credit_score_bayesian(new_data)
    # print(f"Bayesian Predicted Credit Score: {bayes_result['Credit_Score_Bayesian']}")

    combined_result = predict_credit_score_combined(new_data)
    print(f"Combined Predicted Credit Score: {combined_result['Credit_Score']}")
    print("Combined Result:")
    print(combined_result)