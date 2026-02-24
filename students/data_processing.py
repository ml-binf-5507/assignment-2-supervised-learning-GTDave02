"""
Data loading and preprocessing functions for heart disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(filepath):
    """
    Load the heart disease dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the heart disease CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with all features and targets
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV is empty or malformed
        
    Examples
    --------
    >>> df = load_heart_disease_data('data/heart_disease_uci.csv')
    >>> df.shape
    (270, 15)
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("The file could not be located. Please ensure the path is complete and correct.")
    except ValueError:
        print("The file was found but could not be read. Please ensure you have located the correct file and that it is intact.")
    # Hint: Use pd.read_csv()
    # Hint: Check if file exists and raise helpful error if not
    # TODO: Implement data loading
    return df


def preprocess_data(df):
    """
    Handle missing values, encode categorical variables, and clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    # TODO: Implement preprocessing

    data_copy = df.copy()
    # - Handle missing values
    # impute: trestbps, col, thalch, oldpeak
    data_copy["trestbps"].fillna(data_copy["trestbps"].mean(), inplace = True)
    data_copy["col"].fillna(data_copy["col"].mean(), inplace = True)
    data_copy["thalch"].fillna(data_copy["thalch"].mean(), inplace = True)
    data_copy["oldpeak"].fillna(data_copy["oldpeak"].mean(), inplace = True)
    # drop rows: fbs, restecg, exang
    data_dropped = data_copy["fbs"].drop_duplicates()
    data_dropped = data_dropped["restecg"].drop_duplicates()
    data_dropped = data_dropped["exang"].drop_duplicates()
    # drop column: ca, thal
    data_dropped = data_dropped.drop(columns = ["ca", "thal", "slope"])
    # - Encode categorical variables (e.g., sex, cp, fbs, etc.)
    data_encoded = pd.get_dummies(data_dropped, columns = ["sex", "cp", "fbs", "exang", "restecg", "dataset"])
    # - Ensure all columns are numeric
    return data_encoded



def prepare_regression_data(df, target='chol'):
    """
    Prepare data for linear regression (predicting serum cholesterol).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'chol')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector
    """
    # TODO: Implement regression data preparation
    # - Remove rows with missing chol values -> imputed in previous step
    # - Exclude chol from features
    X = df.drop(columns = ["chol"])
    Y = df["chol"]
    # - Return X (features) and y (target)
    return X, Y


def prepare_classification_data(df, target='num'):
    """
    Prepare data for classification (predicting heart disease presence).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'num')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector (binary)
    """
    # TODO: Implement classification data preparation
    # - Binarize target variable
    df["num"] = (df["num"] >= 2).astype(int)
    # - Exclude target from features
    X = df.drop(columns = ["chol", "num"])
    # - Exclude chol from features
    Y = df["num"]
    # - Return X (features) and y (target)
    return X, Y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        where scaler is the fitted StandardScaler
    """
    # TODO: Implement train/test split and scaling
    # - Use train_test_split with provided parameters
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    # - Fit StandardScaler on training data only
    scaler = StandardScaler
    scaler.fit(X_train)
    # - Transform both train and test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # - Return scaled data and scaler object
    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler
