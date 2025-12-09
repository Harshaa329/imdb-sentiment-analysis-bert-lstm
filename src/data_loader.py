"""
data_loader.py
Handles dataset loading and train-test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(file_path)

# Convert target labels to binary
df['sentiment_binary'] = df['sentiment'].map({'negative': 0, 'positive': 1})

def split_train_test(df, test_size=0.2, random_state=42):
    """
    Returns train test split
    Args:
        file_path (str): Path to dataset CSV
        test_size (float): Test set ratio
        random_state (int): Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test (pd.Series)
    """

    X = df['review']
    y = df['sentiment_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

