 📊 End-to-End Telco Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![API](https://img.shields.io/badge/API-FastAPI-green)
![UI](https://img.shields.io/badge/Frontend-Streamlit-red)

## 📌 Project Overview
Customer churn is one of the most significant expenses for telecommunications companies. This project is a complete, end-to-end Machine Learning solution designed to predict which customers are at high risk of canceling their service. 

Instead of just building a model in a notebook, this project features a **fully deployed architecture**:
1. **Machine Learning Pipeline:** A robust Random Forest classifier built with `scikit-learn`.
2. **REST API Backend:** A lightning-fast API built with `FastAPI` to serve real-time predictions.
3. **Interactive Frontend:** A sleek web dashboard built with `Streamlit` for business users to input customer data and instantly view churn risk.
4. **Model Explainability:** SHAP (SHapley Additive exPlanations) integration to ensure the AI's decisions are transparent and interpretable.

## 🏗️ Project Architecture
```text
customer churn ai/
├── api/                   # FastAPI backend server
│   └── main.py            # API endpoints (GET, POST)
├── app.py                 # Streamlit frontend web dashboard
├── data/                  # Raw and processed datasets
├── models/                # Saved pickle files and SHAP visualizations
│   ├── churn_model.pkl    
│   └── shap_summary.png   
├── notebooks/             # Jupyter notebooks for EDA and SHAP explainability
├── src/                   # Training scripts and data pipelines
│   └── train.py           
└── requirements.txt       # Environment dependencies

git clone [https://github.com/your-username/customer-churn-ai.git](https://github.com/your-username/customer-churn-ai.git)
cd "customer churn ai"
pip install -r requirements.txt


The API will be available at http://127.0.0.1:8000. You can view the interactive Swagger UI documentation at http://127.0.0.1:8000/docs.


Leave the API running, open a second terminal, and run:

streamlit run app.py
A browser window will automatically open with the interactive dashboard.

Model Explainability (SHAP)
To ensure the business can trust the model, SHAP values were calculated to determine exactly which features drive customer churn.

(Note: Add your SHAP summary plot image to the models/ folder and it will display here!)

Key business insights discovered:

Contract Type: Month-to-month contracts are the highest driver of churn.

Tenure: Newer customers are significantly more likely to leave than long-term subscribers.

Monthly Charges: Higher monthly charges strongly correlate with a higher risk of cancellation.

author:MD SUHAYL SEKANDER 
