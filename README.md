# Sarcopenia Prediction Using Gradient Boosting Classifier

This project aims to predict **Sarcopenia**, which is the age-related progressive loss of muscle mass and strength, using machine learning. It involves preprocessing medical data, training a machine learning model (Gradient Boosting), and evaluating its performance. The dataset contains both **numerical** and **categorical** features, including data about **Hypertension (HT)** and other related health conditions.


## Project Steps

1. **Decide the Features**:
   - Determine which features are categorical and numerical.
   - For categorical features, certain values might need to be encoded into numerical values.
   - If the feature is an object, it is classified as categorical; otherwise, numerical features are selected.

2. **Splitting by Gender**:
   - The data is split into two datasets: one for **Females** and another for **Males**.

3. **Splitting into Training and Test Sets**:
   - The dataset is split into training and test sets. This is done randomly to ensure that the model can generalize well to unseen data.

4. **Normalization**:
   - Features with different ranges (e.g., `Weight` vs. `Age`) are normalized using **StandardScaler** to bring them into a comparable range for training the model.

## Main Script: `train_and_evaluate.py`

### Overview

This script processes the preprocessed data, trains a **Gradient Boosting** classifier for both female and male subsets, evaluates the model, and saves the evaluation results.

### Steps in the Script

1. **Load Preprocessed Data**: 
   - The script loads preprocessed data for **females** and **males** separately.

2. **Feature and Target Selection**:
   - Select the relevant **numeric** and **categorical** features.
   - Define `Sarc` (Sarcopenia) as the target variable.

3. **Pipeline Creation**:
   - A pipeline is built to preprocess the data (scaling numerical features and one-hot encoding categorical features) and apply a Gradient Boosting Classifier.

4. **Model Training**:
   - The Gradient Boosting Classifier is trained on the training set and evaluated using the test set.

5. **Model Evaluation**:
   - The model is evaluated using metrics such as:
     - **Confusion Matrix**
     - **Classification Report** (precision, recall, F1-score)
     - **Feature Importance** (for interpretability)

6. **Experiment Directory**:
   - Results are stored in an experiment directory with timestamp and metadata.

7. **Saving Results**:
   - The model, confusion matrix, feature importance plot, and evaluation metrics are saved.

8. **Results Storage**:
   - Model evaluation results are saved into an Excel file.

### Results

- The results of the model evaluation, including precision, recall, F1-score, confusion matrix, and feature importance, are saved in an Excel file.

---

## Configuration

The configuration file (`configs/config.yml`) contains the necessary configurations, including:
- Paths to data and experiment directories
- Hyperparameters for the model (e.g., `n_estimators`, `learning_rate`)

## Running the project

Clone the repository:
```bash
git clone https//github.com/Alessio1599/sarcopenia_prediction.git
cd src
```

To preprocess the data, run the following:
```bash
python preprocessing.py
```
To train the model, evaluate it, and save the results, run the following:
```bash
python train_evaluate_model.py
```
This will perform the following:
- Load the preprocessed data
- Train the Gradient Boosting Classifier
- Save the model and evaluation results
- Generate and save the confusion matrix and feature importance plot