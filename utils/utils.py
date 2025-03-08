import os
import pandas as pd
import seaborn as sns
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inspection_dataframe(df, eda_dir="eda"):
    """ Inspect the dataframe and save the results to an Excel file and a PNG file."""
    os.makedirs(eda_dir, exist_ok=True)
    
    row_count=5
    print("\n=== First 5 rows of the dataset ===\n")
    display(df.head(row_count)) # first rows of the dataframe
    df.head(row_count).to_excel(os.path.join(eda_dir, "first_rows.xlsx"), index=False)

    print("\n=== 5 random rows of the dataset ===\n")
    display(df.sample(row_count)) # first rows of the dataframe
    df.sample(row_count).to_excel(os.path.join(eda_dir, "random_rows.xlsx"), index=False)
    

    print("\n=== Brief summary of the dataset ===\n")
    print(df.info()) #brief summary, to check if there are missing values


    print("\n=== Overall statistics of the dataset ===\n")
    stats = df.describe().transpose()
    display(stats)# to see the range of values of the dataframe
    stats.to_excel(os.path.join(eda_dir, "statistics.xlsx")) # Save the statistics to an Excel file


    print("\n=== Histogram of the dataset ===\n")
    df.hist(bins=50, figsize=(18,15)) #Histogram about the values that assume the dataframe
    plt.savefig(os.path.join(eda_dir, "histogram.png"), dpi=300, bbox_inches='tight')  # Save the figure
    plt.close() # plt.show()
    
    logging.info("Inspection of the dataframe completed")
 
def print_unique_values(df):
    """
    Loop through columns and print unique values with their counts for columns with less than 20 unique values.
    """
    for col in df.columns:
        if df[col].nunique() < 20:
            print(f"\nUnique values and their counts in '{col}':")
            print(df[col].value_counts())    
    
def save_confusion_matrix_image(conf_matrix, class_names, output_path, figsize=(10, 10)):
    """Saves the confusion matrix as an image."""
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.matshow(conf_matrix, cmap=sns.color_palette("crest", as_cmap=True)) 
    plt.colorbar(img)  # Add color scale
    
    tick_marks = np.arange(len(class_names))
    _ = plt.xticks(tick_marks, class_names, rotation=45)
    _ = plt.yticks(tick_marks, class_names)
    _ = plt.ylabel('Real')
    _ = plt.xlabel('Predicted')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, '{0:.1%}'.format(conf_matrix[i, j]),
                           ha='center', va='center', color='w')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ Confusion matrix image saved to {output_path}.")
    
def plot_feature_importance(importances, feature_names, model_name, output_dir):
    """Plots and saves feature importance for a given model."""
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    df = df.sort_values('Importance')
    
    plt.figure()
    
    # Create horizontal bar plot
    sns.barplot(data=df,
                x='Importance',
                y='Feature',
                hue='Feature',
                palette='crest', #viridis
                legend=False)
    
    # Customize the plot
    plt.xlabel('Importance score')
    plt.ylabel('Feature')
    
    # Save the figure
    feature_importance_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_feature_importance.png")
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"✅Feature importance plot saved to {feature_importance_path}.")

    return feature_importance_path