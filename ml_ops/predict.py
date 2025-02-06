
import joblib
import pandas as pd


# Predict function for new account
def predict_lockin_score(new_account):
    model = joblib.load("cloud_lockin_model.pkl")
    new_df = pd.DataFrame([new_account])
    predicted_score = model.predict(new_df)[0]
    return round(predicted_score, 2)


if __name__ == "__main__":
    # Example usage
    new_account = {
        "AWS EKS": 100,
        "AWS RDS Postgres": 2,
        "AWS RDS Aurora": 0,
        "AWS Step Functions": 100,
        "AWS Lambda Container": 50,
        "AWS Lambda Code": 100,
        "AWS S3": 10,
        "AWS DynamoDB": 0,
        "AWS EC2": 8,
        "AWS CloudFormation": 0,
        "AWS SageMaker": 0,
        "AWS Redshift": 0,
        "AWS Elastic Beanstalk": 0,
        "AWS API Gateway": 100,
    }
    print(f"Predicted Cloud Lock-in Score: {predict_lockin_score(new_account)}")