# src/preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

FILE_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(FILE_DIR)
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

from loguru import logger
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format: '2025-03-07_14-30-45'

logger.add(os.path.join(LOGS_DIR, f'preprocessing_{timestamp}.log'))

import sys
sys.path.append(BASE_DIR)

from configs.config import cfg
from utils.utils import inspection_dataframe

#logging.basicConfig(level=logging.INFO)


def load_raw_data(filepath):
    """ Load raw data from the specified file path."""
    df = pd.read_csv(filepath)
    logger.info(f"Raw data loaded successfully from {filepath}. Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def remove_leading_trailing_spaces(df):
    """ Remove leading and trailing spaces from the column names."""
    df.columns = df.columns.str.strip()
    logger.info("Loading and trailing spaces removed from the column names.")
    return df

def remove_missing_values(df):
    """ Remove rows with missing values and log the number of rows removed."""
    initial_shape = df.shape[0]
    df = df.dropna()
    final_shape = df.shape[0]
    rows_removed = initial_shape - final_shape
    logger.info(f"Rows with missing values removed: {rows_removed}. Dataset Shape: {final_shape} rows and {df.shape[1]} columns.")
    return df

def encode_cat_features(df):
    """ Encode categorical features."""
    #  Encode 'Exercise'
    exercise_map = {
        '0': 0,
        '1-2/week': 1,
        '1-2 hf': 1,
        '3-4/week': 2,
        '3-4 hf': 2
    }
    df['Exercise'] = df['Exercise'].map(exercise_map).astype(int)

    # Encode 'Education'
    education_map = {
        'illiterate': 0,
        'okuryazar değil': 0,
        'Primary School': 1,
        'primary school': 1,
        'ilkokul': 1,
        'Primary shool': 1,
        'Secondary School': 2,
        'ortaokul': 2,
        'High School': 3,
        'high school': 3,
        'lise': 3,
        'University': 4,
        'university': 4,
        'üniversite': 4
    }
    df['Education'] = df['Education'].map(education_map).astype(int)

    # Encode 'Smoking': 0 -> Non-smoker, 1 -> Smoker
    df.loc[:,'Smoking'] = df['Smoking'].astype(int)
    
    # Encode 'Gender': F -> 0, M -> 1
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    
    logger.info("Categorical features encoded successfully.")
    return df

def split_by_gender(df):
    """ Split the dataset into two separate datasets based on gender."""
    df_female = df[df['Gender'] == 0]
    df_male = df[df['Gender'] == 1]
    return df_female, df_male


def main():
    filepath = cfg['data']['raw_root'] + '/Sampled_Sarcopenia_Data.csv'
    df = load_raw_data(filepath)
    
    #inspection_dataframe(df)
    
    df = remove_leading_trailing_spaces(df)
    
    # Select relevant features for modeling
    selected_features = ['Age', 'Gender', 'Weight', 'Height', 'DM', 'CST', 'HT', 
                     'BMI', 'Exercise', 'Education', 'Smoking', 'STAR','Grip Strength', 'Sarcopenia']

    df = df[selected_features]
    
    df = remove_missing_values(df)
    
    df = encode_cat_features(df)
    
    df = df.rename(columns={'Grip Strength': "HGS",'Sarcopenia': 'Sarc'}) #rename 'Grip Strength' to 'HGS' and 'Sarcopenia' to 'Sarc'
    
    #  Ensure 'DM', 'HT', 'Sarcopenia' are integers
    df.loc[:,'DM'] = df['DM'].astype(int)
    df.loc[:,'HT'] = df['HT'].astype(int)
    df.loc[:,'Sarc'] = df['Sarc'].astype(int)
    
    df_female, df_male = split_by_gender(df)
    
    # Save preprocessed data
    output_path = cfg['data']['preprocessed_root']
    df_female.to_csv(f"{output_path}/preprocessed_female.csv", index=False)
    df_male.to_csv(f"{output_path}/preprocessed_male.csv", index=False)
    
    logger.info(f"Preprocessed data saved successfully in {output_path}.")


if __name__ == '__main__':
    main()