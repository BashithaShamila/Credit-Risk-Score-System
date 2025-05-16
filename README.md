# Credit Risk Score Prediction System

A machine learning system that predicts credit risk scores using XGBoost and Bayesian Network models.

## Overview

This project implements a credit risk assessment tool that combines multiple machine learning approaches:
- XGBoost classification model
- Bayesian Network probabilistic model
- Combined prediction using both models

The system provides credit score predictions based on financial and demographic attributes, which can be used in risk assessment for lending decisions.

## Features

- Multi-model prediction approach (XGBoost + Bayesian Networks)
- Docker containerization for easy deployment
- Pre-trained models included
- Support for various loan types in prediction

## Technical Stack

- Python 3.10
- XGBoost 2.1.4
- pandas 2.2.2
- scikit-learn 1.6.1
- pgmpy 0.1.21 (for Bayesian network)
- Docker

## Project Structure

```
.
├── Dockerfile                # Docker configuration
├── predict_credit_score.py   # Main prediction script
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── models/                   # Pre-trained models
    ├── age_discretizer.pkl           # Discretizer for age values
    ├── bayes_model.pkl               # Trained Bayesian network model
    ├── Credit_Score_label_encoder.pkl # Label encoder for credit scores
    ├── income_discretizer.pkl        # Discretizer for income values
    ├── Occupation_label_encoder.pkl  # Label encoder for occupations
    └── xgb_model.json                # Trained XGBoost model
```

## Installation

### Local Installation

1. Clone this repository:
   ```
   git clone <your-repo-url>
   cd credit-risk-score
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Docker Installation

Build and run the Docker container:

```
docker build -t credit-risk-score .
docker run credit-risk-score
```

## Usage

### From Python

```python
from predict_credit_score import predict_credit_score_combined

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

# Get prediction
result = predict_credit_score_combined(new_data)
print(f"Predicted Credit Score: {result['Credit_Score']}")
```

## Model Information

This project uses two complementary models:

1. **XGBoost Classifier**: A gradient boosting decision tree model that excels at handling numerical features and complex nonlinear relationships.

2. **Bayesian Network**: A probabilistic graphical model that captures causal relationships between variables.

The final prediction is calculated by averaging the predictions from both models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Contact

[Add your contact information here]