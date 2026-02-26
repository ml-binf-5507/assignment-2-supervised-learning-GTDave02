"""
Linear regression functions for predicting cholesterol using ElasticNet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score


def train_elasticnet_grid(X_train, y_train, l1_ratios, alphas):
    """
    Train ElasticNet models over a grid of hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix
    y_train : np.ndarray or pd.Series
        Training target vector
    l1_ratios : list or np.ndarray
        L1 ratio values to test (0 = L2 only, 1 = L1 only)
    alphas : list or np.ndarray
        Regularization strength values to test
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['l1_ratio', 'alpha', 'r2_score', 'model']
        Contains R² scores for each parameter combination on training data
    """
    # TODO: Implement grid search
    # - Create results list
    # - For each combination of l1_ratio and alpha:
    #   - Train ElasticNet model with max_iter=5000

    r2_value_list = []
    model_list = []
    ratio_list = []
    alpha_list = []

    for ratio in l1_ratios:
        for alpha in alphas:
            model = ElasticNet(l1_ratio=ratio, alpha=alpha, max_iter=5000)
            model.fit(X_train, y_train)
            ratio_list.append(ratio)
            alpha_list.append(alpha)
            model_list.append(model)
    #   - Calculate R² score on training data
            r2_value_list.append(r2_score(X_train, y_train))
    #   - Store results
    results_df = {
        "l1_ratio":ratio_list,
        "alpha":alpha_list,
        "r2_score":r2_value_list,
        "model":model_list}

    results_df = pd.DataFrame(results_df)
    # - Return DataFrame with results
    return results_df


def create_r2_heatmap(results_df, l1_ratios, alphas, output_path=None):
    """
    Create a heatmap of R² scores across l1_ratio and alpha parameters.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from train_elasticnet_grid
    l1_ratios : list or np.ndarray
        L1 ratio values used in grid
    alphas : list or np.ndarray
        Alpha values used in grid
    output_path : str, optional
        Path to save figure. If None, returns figure object
        
    Returns
    -------
    matplotlib.figure.Figure
        The heatmap figure
    """
    # TODO: Implement heatmap creation
    # - Pivot results_df to create matrix with l1_ratio on x-axis, alpha on y-axis
    results_matrix = results_df.pivot(index = "alpha", columns = "l1_ratio", values = "r2_score")
    # - Create heatmap using seaborn
    sns.heatmap(results_matrix)
    # - Set labels: "L1 Ratio", "Alpha", "R² Score"
    plt.xlabel("L1 Ratio")
    plt.ylabel("Alpha")
    plt.title("ElasticNet R-Squared Scores")
    # - Add colorbar
    # - Save to output_path if provided
    if output_path != None:
        plt.savefig(output_path)
    # - Return figure object
    return plt


def get_best_elasticnet_model(X_train, y_train, X_test, y_test, 
                               l1_ratios=None, alphas=None):
    """
    Find and train the best ElasticNet model on test data.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray or pd.Series
        Test target
    l1_ratios : list, optional
        L1 ratio values to test. Default: [0.1, 0.3, 0.5, 0.7, 0.9]
    alphas : list, optional
        Alpha values to test. Default: [0.001, 0.01, 0.1, 1.0, 10.0]
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': fitted ElasticNet model
        - 'best_l1_ratio': best l1 ratio
        - 'best_alpha': best alpha
        - 'train_r2': R² on training data
        - 'test_r2': R² on test data
        - 'results_df': full results DataFrame
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    # TODO: Implement best model selection
    # - Train models using train_elasticnet_grid
    elastic_net_train = train_elasticnet_grid(X_train, y_train)
    # - Select model with highest test R² (not training R²)
    r2_test_list = []
    for model in elastic_net_train["model"]:
        r2_test = model.score(X_test, y_test)
        r2_test_list.append(r2_test)
    elastic_net_train["test_r2"] = r2_test_list
    best_model = elastic_net_train.loc[elastic_net_train["test_r2"].idxmax()]
    # - Return dictionary with best model and parameters

    best_params = {
        "model":elastic_net_train.iloc[best_model]["model"], 
        "best_l1_ratio":elastic_net_train.iloc[best_model]["l1_ratio"],
        "best_alpha":elastic_net_train.iloc[best_model]["alpha"],
        "train_r2":elastic_net_train.iloc[best_model]["r2_score"],
        "test_r2":elastic_net_train.iloc[best_model]["test_r2"],
        "results_df":elastic_net_train
        }
    return best_params
