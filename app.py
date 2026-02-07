import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import pickle
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🩺",
    layout="wide"
)

# Title and description
st.title("🩺 Breast Cancer Classification Dashboard")
st.markdown("""
This application implements 6 different machine learning models for breast cancer classification.
The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset from UCI.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the mode",
    ["Home", "Dataset Overview", "Train Models", "Model Evaluation", "Make Predictions"]
)

# Load dataset
@st.cache_data
def load_data():
    # For demo purposes, we'll load the dataset from sklearn
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].apply(lambda x: 'Malignant' if x == 0 else 'Benign')
    return df, data

df, data_obj = load_data()

# Home Page
if app_mode == "Home":
    st.header("Welcome to the Breast Cancer Classification System")
    st.markdown("""
    ### 📋 Assignment Requirements Implemented:
    
    1. **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset
       - Features: 30 numerical features
       - Instances: 569 samples
       - Classes: 2 (Malignant/Benign)
    
    2. **Models Implemented**:
       - Logistic Regression
       - Decision Tree Classifier
       - K-Nearest Neighbors
       - Gaussian Naive Bayes
       - Random Forest (Ensemble)
       - XGBoost (Ensemble)
    
    3. **Evaluation Metrics**:
       - Accuracy
       - AUC Score
       - Precision
       - Recall
       - F1 Score
       - Matthews Correlation Coefficient
    """)
    
    st.info("💡 Use the sidebar to navigate through different sections")

# Dataset Overview
elif app_mode == "Dataset Overview":
    st.header("📊 Dataset Overview")
    
    # Show dataset info
    st.subheader("Dataset Shape")
    st.write(f"Number of samples: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1] - 2}")  # Excluding target and diagnosis
    
    # Show class distribution
    st.subheader("Class Distribution")
    class_dist = df['diagnosis'].value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar plot
    class_dist.plot(kind='bar', ax=ax[0], color=['lightgreen', 'salmon'])
    ax[0].set_title('Class Distribution')
    ax[0].set_xlabel('Diagnosis')
    ax[0].set_ylabel('Count')
    
    # Pie chart
    ax[1].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', 
              colors=['lightgreen', 'salmon'], startangle=90)
    ax[1].set_title('Class Proportions')
    
    st.pyplot(fig)
    
    # Show sample data
    st.subheader("Sample Data (First 10 rows)")
    st.dataframe(df.head(10))
    
    # Show feature statistics
    st.subheader("Feature Statistics")
    st.dataframe(df.describe())

# Train Models Section
elif app_mode == "Train Models":
    st.header("🚀 Train Machine Learning Models")
    
    # Data preprocessing
    st.subheader("Data Preprocessing")
    
    # Select features and target
    X = df.drop(['target', 'diagnosis'], axis=1)
    y = df['target']
    
    # Split the data
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.write(f"Training set size: {X_train.shape[0]} samples")
    st.write(f"Test set size: {X_test.shape[0]} samples")
    
    # Model selection
    st.subheader("Select Models to Train")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        train_lr = st.checkbox("Logistic Regression", value=True)
        train_dt = st.checkbox("Decision Tree", value=True)
    with col2:
        train_knn = st.checkbox("K-Nearest Neighbors", value=True)
        train_nb = st.checkbox("Naive Bayes", value=True)
    with col3:
        train_rf = st.checkbox("Random Forest", value=True)
        train_xgb = st.checkbox("XGBoost", value=True)
    
    if st.button("Train Selected Models"):
        models = {}
        results = []
        
        with st.spinner("Training models..."):
            # Initialize progress bar
            progress_bar = st.progress(0)
            model_count = sum([train_lr, train_dt, train_knn, train_nb, train_rf, train_xgb])
            
            current_model = 0
            
            # Logistic Regression
            if train_lr:
                st.write("Training Logistic Regression...")
                lr = LogisticRegression(random_state=42, max_iter=1000)
                lr.fit(X_train_scaled, y_train)
                models['Logistic Regression'] = lr
                current_model += 1
                progress_bar.progress(current_model / model_count)
            
            # Decision Tree
            if train_dt:
                st.write("Training Decision Tree...")
                dt = DecisionTreeClassifier(random_state=42, max_depth=5)
                dt.fit(X_train, y_train)
                models['Decision Tree'] = dt
                current_model += 1
                progress_bar.progress(current_model / model_count)
            
            # K-Nearest Neighbors
            if train_knn:
                st.write("Training K-Nearest Neighbors...")
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train_scaled, y_train)
                models['K-Nearest Neighbors'] = knn
                current_model += 1
                progress_bar.progress(current_model / model_count)
            
            # Naive Bayes
            if train_nb:
                st.write("Training Naive Bayes...")
                nb = GaussianNB()
                nb.fit(X_train, y_train)
                models['Naive Bayes'] = nb
                current_model += 1
                progress_bar.progress(current_model / model_count)
            
            # Random Forest
            if train_rf:
                st.write("Training Random Forest...")
                rf = RandomForestClassifier(random_state=42, n_estimators=100)
                rf.fit(X_train, y_train)
                models['Random Forest'] = rf
                current_model += 1
                progress_bar.progress(current_model / model_count)
            
            # XGBoost
            if train_xgb:
                st.write("Training XGBoost...")
                xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=100)
                xgb_model.fit(X_train, y_train)
                models['XGBoost'] = xgb_model
                current_model += 1
                progress_bar.progress(current_model / model_count)
        
        # Evaluate all trained models
        st.subheader("Model Evaluation Results")
        
        for name, model in models.items():
            # Predict
            if name in ['Logistic Regression', 'K-Nearest Neighbors']:
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            results.append({
                'Model': name,
                'Accuracy': f"{accuracy:.4f}",
                'AUC': f"{auc:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1 Score': f"{f1:.4f}",
                'MCC': f"{mcc:.4f}"
            })
            
            # Save the model
            if name in ['Logistic Regression', 'K-Nearest Neighbors']:
                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'requires_scaling': True
                }
            else:
                model_data = {
                    'model': model,
                    'scaler': None,
                    'requires_scaling': False
                }
            
            joblib.dump(model_data, f"model/{name.replace(' ', '_').lower()}.pkl")
        
        # Display results table
        results_df = pd.DataFrame(results)
        st.table(results_df)
        
        # Save results to CSV
        results_df.to_csv("model_evaluation_results.csv", index=False)
        st.success("✅ Models trained and saved successfully!")
        
        # Save scaler
        joblib.dump(scaler, "model/scaler.pkl")

# Model Evaluation Section
elif app_mode == "Model Evaluation":
    st.header("📈 Model Evaluation and Comparison")
    
    # Load pre-trained models if available
    model_files = [f for f in os.listdir("model") if f.endswith(".pkl") and f != "scaler.pkl"]
    
    if not model_files:
        st.warning("No trained models found. Please train models first.")
    else:
        # Load all models
        models = {}
        results = []
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            try:
                model_data = joblib.load(f"model/{model_file}")
                models[model_name] = model_data
                st.success(f"✅ Loaded {model_name}")
            except:
                st.error(f"❌ Failed to load {model_name}")
        
        # Load test data
        X = df.drop(['target', 'diagnosis'], axis=1)
        y = df['target']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load scaler
        scaler = joblib.load("model/scaler.pkl")
        
        # Evaluate each model
        for name, model_data in models.items():
            model = model_data['model']
            requires_scaling = model_data.get('requires_scaling', False)
            
            # Prepare test data
            if requires_scaling:
                X_test_processed = scaler.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate metrics
            metrics = {
                'Model': name,
                'Accuracy': f"{accuracy_score(y_test, y_pred):.4f}",
                'AUC': f"{roc_auc_score(y_test, y_pred_proba):.4f}",
                'Precision': f"{precision_score(y_test, y_pred):.4f}",
                'Recall': f"{recall_score(y_test, y_pred):.4f}",
                'F1 Score': f"{f1_score(y_test, y_pred):.4f}",
                'MCC': f"{matthews_corrcoef(y_test, y_pred):.4f}"
            }
            results.append(metrics)
            
            # Display confusion matrix for selected model
            if st.checkbox(f"Show confusion matrix for {name}", key=f"cm_{name}"):
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Confusion Matrix - {name}')
                st.pyplot(fig)
                
                # Classification report
                st.text(f"Classification Report for {name}:")
                report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])
                st.text(report)
        
        # Display comparison table
        st.subheader("📊 Model Performance Comparison")
        results_df = pd.DataFrame(results)
        st.table(results_df)
        
        # Visualization of metrics
        st.subheader("📈 Performance Metrics Visualization")
        
        # Convert string metrics to float for plotting
        plot_df = results_df.copy()
        for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']:
            plot_df[col] = plot_df[col].astype(float)
        
        # Select metrics to visualize
        metrics_to_plot = st.multiselect(
            "Select metrics to visualize",
            ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
            default=['Accuracy', 'F1 Score', 'AUC']
        )
        
        if metrics_to_plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=ax)
            ax.set_title('Model Performance Comparison')
            ax.set_ylabel('Score')
            ax.set_xlabel('Model')
            ax.legend(title='Metrics')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# Make Predictions Section
elif app_mode == "Make Predictions":
    st.header("🔍 Make Predictions on New Data")
    
    st.info("""
    💡 Upload a CSV file with test data for prediction. 
    The file should contain the same 30 features as the training data.
    """)
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load the data
            test_df = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(test_df.head())
            
            st.write(f"Shape of uploaded data: {test_df.shape}")
            
            # Check if all required features are present
            required_features = df.drop(['target', 'diagnosis'], axis=1).columns.tolist()
            missing_features = [feat for feat in required_features if feat not in test_df.columns]
            
            if missing_features:
                st.error(f"Missing features: {missing_features}")
            else:
                # Select model for prediction
                model_files = [f for f in os.listdir("model") if f.endswith(".pkl") and f != "scaler.pkl"]
                
                if not model_files:
                    st.warning("No trained models found. Please train models first.")
                else:
                    model_names = [f.replace('.pkl', '').replace('_', ' ').title() for f in model_files]
                    selected_model = st.selectbox("Select model for prediction", model_names)
                    
                    if st.button("Make Predictions"):
                        # Load selected model
                        model_file = selected_model.lower().replace(' ', '_') + '.pkl'
                        model_data = joblib.load(f"model/{model_file}")
                        model = model_data['model']
                        requires_scaling = model_data.get('requires_scaling', False)
                        
                        # Load scaler if needed
                        if requires_scaling:
                            scaler = joblib.load("model/scaler.pkl")
                            X_test = scaler.transform(test_df[required_features])
                        else:
                            X_test = test_df[required_features]
                        
                        # Make predictions
                        predictions = model.predict(X_test)
                        probabilities = model.predict_proba(X_test)
                        
                        # Create results dataframe
                        results_df = test_df.copy()
                        results_df['Prediction'] = predictions
                        results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'Malignant', 1: 'Benign'})
                        results_df['Probability_Malignant'] = probabilities[:, 0]
                        results_df['Probability_Benign'] = probabilities[:, 1]
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df[['Prediction_Label', 'Probability_Malignant', 'Probability_Benign']].head())
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Show prediction distribution
                        st.subheader("Prediction Distribution")
                        pred_dist = results_df['Prediction_Label'].value_counts()
                        
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        pred_dist.plot(kind='bar', ax=ax[0], color=['salmon', 'lightgreen'])
                        ax[0].set_title('Prediction Distribution')
                        ax[0].set_xlabel('Diagnosis')
                        ax[0].set_ylabel('Count')
                        
                        ax[1].pie(pred_dist.values, labels=pred_dist.index, autopct='%1.1f%%',
                                 colors=['salmon', 'lightgreen'], startangle=90)
                        ax[1].set_title('Prediction Proportions')
                        
                        st.pyplot(fig)
                        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to make predictions")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Assignment Info")
st.sidebar.info("""
**M.Tech (AIML/DSE)**  
**Machine Learning - Assignment 2**  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Models:** 6 Classification Algorithms

""")
