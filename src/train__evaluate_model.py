# src/train_evaluate_model.py
import os
import pandas as pd
import json
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

FILE_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(FILE_DIR)
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

from loguru import logger
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format: '2025-03-07_14-30-45'

logger.add(os.path.join(LOGS_DIR, f'train_evaluate{timestamp}.log'))

import sys
sys.path.append(BASE_DIR)

from configs.config import cfg
from utils.utils import save_confusion_matrix_image, plot_feature_importance

#logging.basicConfig(level=logging.INFO)

def load_preprocessed_data(filepath):
    data = pd.read_csv(filepath)
    filename = os.path.basename(filepath)
    logger.info(f"✅Preprocessed data {filename} loaded successfully. Dataset Shape: {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

def train_model(X_train, y_train):
    """ Train a Gradient Boosting Classifier. """
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def build_pipeline(model_config, numeric_features, categorical_features):
    """ Build a pipeline using the provided model configuration. """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    classifier = GradientBoostingClassifier(**model_config)  # Load config dynamically
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline

def create_experiment_dir():
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(cfg["experiments_root"], f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger.info(f"✅ Experiment directory created: {experiment_dir}")
    return experiment_dir

def save_experiment_metadata(cfg, experiment_dir):
    """Saves experiment metadata."""
    metadata = {
        "config": cfg
    }
    with open(os.path.join(experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(metadata, f, indent=4)

def evaluate_model(model, X_test, y_test, label, experiment_dir, feature_names):
    """Evaluate the model, save confusion matrix, and plot feature importance."""
    y_pred = model.predict(X_test)

    # Compute Metrics
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    # Save Confusion Matrix
    cm_image_path = os.path.join(experiment_dir, f"{label.lower()}_confusion_matrix.png")
    save_confusion_matrix_image(cm, ["No Sarcopenia", "Sarcopenia"], cm_image_path)

    # Extract Feature Importance (from classifier inside pipeline)
    classifier = model.named_steps["classifier"]  # Extract GradientBoosting model
    importances = classifier.feature_importances_

    # Plot and save feature importance
    feature_importance_path = plot_feature_importance(importances, feature_names, label, experiment_dir)

    # Structure results
    results = {
        "Model": f"Gradient Boosting ({label})",
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": cm.tolist(),
        "Feature Importance": dict(zip(feature_names, importances)),  # Store as dictionary
        "Classification Report": report,
        "Confusion Matrix Path": cm_image_path,
        "Feature Importance Path": feature_importance_path
    }

    return results



def save_results_to_excel(results, experiment_dir):
    """Saves model evaluation results to an Excel file."""
    results_file = os.path.join(cfg["experiments_root"], "experiment_results.xlsx")

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Check if file exists -> Append or Create New
    if os.path.exists(results_file):
        existing_df = pd.read_excel(results_file)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(results_file, index=False)

    logger.info(f"✅Results (precision, recall, F1 score, confusion matrix, feature importance) saved to {results_file}")

def main():
    # Load datasets
    df_female = load_preprocessed_data(cfg['data']['preprocessed_root'] + '/preprocessed_female.csv')
    df_male = load_preprocessed_data(cfg['data']['preprocessed_root'] + '/preprocessed_male.csv')

    # Load preprocessing parameters
    test_size = cfg["preprocessing"]["test_size"]
    random_state = cfg["preprocessing"]["random_state"]

    # Define feature sets
    numeric_features = ['Age', 'Weight', 'Height', 'CST', 'HGS', 'BMI']
    categorical_features = ['DM', 'HT', 'Exercise', 'Education', 'Smoking']
    target = 'Sarc'

    # Create experiment directory
    experiment_dir = create_experiment_dir()

    # Save experiment metadata
    save_experiment_metadata(cfg, experiment_dir)

    # Train and evaluate models for Female & Male datasets
    results = []

    for gender, df in zip(["Female", "Male"], [df_female, df_male]):
        X = df[numeric_features + categorical_features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Build pipeline
        model_config = cfg['models']['gradient_boosting']
        pipeline = build_pipeline(model_config, numeric_features, categorical_features)

        # Train the model
        pipeline.fit(X_train, y_train)
        logger.info(f"✅ Gradient Boosting model trained for {gender} patients.")

        # Save trained model
        model_path = os.path.join(experiment_dir, f"gb_{gender.lower()}.pkl")
        joblib.dump(pipeline, model_path)
        logger.info(f"✅ {gender} model saved at {model_path}")

        # Get One-Hot Encoded Feature Names Correctly
        cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        categorical_features_encoded = list(cat_encoder.get_feature_names_out(categorical_features))
        
        # Combine numerical and categorical feature names
        all_feature_names = numeric_features + categorical_features_encoded
        
        # Evaluate the model
        result = evaluate_model(pipeline, X_test, y_test, gender, experiment_dir, all_feature_names)
        results.append(result)

    # Save results
    save_results_to_excel(results, experiment_dir)

    

if __name__ == '__main__':
    main()