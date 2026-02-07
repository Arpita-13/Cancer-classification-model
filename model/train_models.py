"""
Model Training Script for Breast Cancer Classification
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def load_and_prepare_data():
    """Load and prepare the breast cancer dataset"""
    print("Loading dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': data.feature_names
    }

def train_models(data):
    """Train all 6 classification models"""
    print("\nTraining models...")
    
    models = {}
    results = []
    
    # 1. Logistic Regression
    print("1. Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(data['X_train_scaled'], data['y_train'])
    models['Logistic Regression'] = {
        'model': lr,
        'requires_scaling': True
    }
    
    # 2. Decision Tree
    print("2. Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(data['X_train'], data['y_train'])
    models['Decision Tree'] = {
        'model': dt,
        'requires_scaling': False
    }
    
    # 3. K-Nearest Neighbors
    print("3. Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(data['X_train_scaled'], data['y_train'])
    models['K-Nearest Neighbors'] = {
        'model': knn,
        'requires_scaling': True
    }
    
    # 4. Naive Bayes
    print("4. Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(data['X_train'], data['y_train'])
    models['Naive Bayes'] = {
        'model': nb,
        'requires_scaling': False
    }
    
    # 5. Random Forest
    print("5. Training Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(data['X_train'], data['y_train'])
    models['Random Forest'] = {
        'model': rf,
        'requires_scaling': False
    }
    
    # 6. XGBoost
    print("6. Training XGBoost...")
    xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=100)
    xgb_model.fit(data['X_train'], data['y_train'])
    models['XGBoost'] = {
        'model': xgb_model,
        'requires_scaling': False
    }
    
    # Evaluate all models
    print("\nEvaluating models...")
    for name, model_info in models.items():
        model = model_info['model']
        requires_scaling = model_info['requires_scaling']
        
        # Prepare test data
        if requires_scaling:
            X_test = data['X_test_scaled']
        else:
            X_test = data['X_test']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(data['y_test'], y_pred),
            'AUC': roc_auc_score(data['y_test'], y_pred_proba),
            'Precision': precision_score(data['y_test'], y_pred),
            'Recall': recall_score(data['y_test'], y_pred),
            'F1': f1_score(data['y_test'], y_pred),
            'MCC': matthews_corrcoef(data['y_test'], y_pred)
        }
        results.append(metrics)
        
        # Save the model
        joblib.dump(model_info, f"C:/Users/arpit/Desktop/BITS_PILANI/SEM 2/MACHINE LEARNING/cancer classification analysis/model/{name.lower().replace(' ', '_')}.pkl")
        print(f" {name} saved")
    
    # Save scaler
    joblib.dump(data['scaler'], "C:/Users/arpit/Desktop/BITS_PILANI/SEM 2/MACHINE LEARNING/cancer classification analysis/model/scaler.pkl")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv("C:/Users/arpit/Desktop/BITS_PILANI/SEM 2/MACHINE LEARNING/cancer classification analysis/model/model_evaluation_results.csv", index=False)
    
    return results_df, models

def main():
    """Main function to train and evaluate all models"""
    print("=" * 60)
    print("BREAST CANCER CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Train models
    results_df, models = train_models(data)
    
    # Display results
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE RESULTS")
    print("=" * 60)
    print("\nEvaluation Metrics:")
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("BEST PERFORMING MODEL BY METRIC:")
    print("=" * 60)
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    for metric in metrics:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"{metric}: {best_model} ({best_score:.4f})")
    
    print("\n" + "=" * 60)
    print("All models have been trained and saved in the 'model' folder")
    print("=" * 60)

if __name__ == "__main__":
    main()