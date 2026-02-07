# Cancer-classification-model
Problem Statement:
-This project implements and compares six machine learning models to classify breast tumors as malignant or benign using the Breast Cancer Wisconsin Diagnostic Dataset. The   goal is to accurately predict tumor diagnosis based on 30 cytological features extracted from digitized images.

Dataset Description:
-Source: UCI Machine Learning Repository
-Instances: 569 samples
-Features: 30 numerical measurements including mean radius, texture, perimeter, area, and smoothness
-Target: Binary classification (Malignant=0, Benign=1)
-Class Distribution: 357 benign (62.7%), 212 malignant (37.3%)

Models & Performance
-Six classification models were implemented and evaluated: Logistic regression, decision tree, k-nearest neighbours, naive bayes, random forest, xgboost

Key Observations
-XGBoost performed best overall with highest accuracy (97.37%) and MCC score (0.9446)
-Random Forest achieved the highest AUC (0.9967), indicating excellent class discrimination
-All models exceeded 93% accuracy, demonstrating the dataset's predictability
-Ensemble methods (XGBoost, Random Forest) outperformed individual classifiers
-Feature scaling improved performance for Logistic Regression and KNN models

Project Features
-Interactive Streamlit dashboard for model comparison
-CSV upload functionality for test predictions

Visualization of confusion matrices and classification reports
-Performance metrics comparison across all six models
-Downloadable prediction results

Technologies Used
-Python, Scikit-learn, XGBoost
-Streamlit for web deployment
-Pandas, NumPy for data processing
-Matplotlib, Seaborn for visualizations

Setup & Deployment
-Install dependencies: pip install -r requirements.txt
-Train models: python model/train_models.py
-Run locally: streamlit run app.py

Deploy to Streamlit Cloud via GitHub repository
