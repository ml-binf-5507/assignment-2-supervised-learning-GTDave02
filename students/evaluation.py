"""
Model evaluation functions: metrics and ROC/PR curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, auc as compute_auc, r2_score
)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score for regression.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True target values
    y_pred : np.ndarray or pd.Series
        Predicted target values
        
    Returns
    -------
    float
        R² score (between -inf and 1, higher is better)
    """
    # TODO: Implement R² calculation
    # Use sklearn's r2_score
    r2_value = r2_score(y_true, y_pred)
    return r2_value


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred : np.ndarray or pd.Series
        Predicted binary labels
        
    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # TODO: Implement metrics calculation
    # Return dictionary with all four metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1
    }

    return metrics


def calculate_auroc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the ROC Curve (AUROC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUROC score (between 0 and 1)
    """
    # TODO: Implement AUROC calculation
    # Use sklearn's roc_auc_score
    roc_score = roc_auc_score(y_true, y_pred_proba)
    return roc_score


def calculate_auprc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUPRC score (between 0 and 1)
    """
    # TODO: Implement AUPRC calculation
    # Use sklearn's average_precision_score
    aps = average_precision_score(y_true, y_pred_proba)
    return aps


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", 
                        output_path=None, ax=None):
    """
    Generate and plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement ROC curve plotting
    # - Calculate ROC curve using roc_curve()
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    # - Calculate AUROC using auc()
    AUROC_curve = auc(fpr, tpr)
    # - Plot curve with label showing AUROC score
    plt.figure()
    plt.plot(fpr, tpr, color = "orange", lw = 2, label = "ROC curve (area = %0.2f)" % AUROC_curve)
    # - Add diagonal reference line
    plt.plot([0,1], [0,1], color = "navy", lw = 2, linestyle = "--")
    # - Set labels: "False Positive Rate", "True Positive Rate"
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(title = model_name, loc = "lower right")
    # - Save to output_path if provided
    if output_path != None:
        plt.savefig(output_path)
    # - Return figure and/or axes
    if ax == None:
        return figure
    elif ax != None:
        return figure, ax


def generate_auprc_curve(y_true, y_pred_proba, model_name="Model",
                        output_path=None, ax=None):
    """
    Generate and plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement PR curve plotting
    # - Calculate precision-recall curve using precision_recall_curve()
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    # - Calculate AUPRC using average_precision_score()
    average_precision = average_precision_score(y_true, y_pred_proba)
    # - Plot curve with label showing AUPRC score
    plt.figure()
    plt.plot(recall, precision, color = "blue", lw = 2, label = "Precision-Recall Curve (area = %0.2f)" % average_precision)
    # - Add horizontal baseline (prevalence)
    prevalence = y_pred_proba.mean()
    plt.plot(prevalence, color = "red", lw = 2, linestyle = "--", label = "Baseline (Prevalence)")
    # - Set labels: "Recall", "Precision"
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(title = model_name, loc = "lower right)")
    # - Save to output_path if provided
    if output_path != None:
        plt.savefig(output_path)
    # - Return figure and/or axes
    if ax == None:
        return figure
    elif ax != None:
        return figure, ax


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn,
                          output_path=None):
    """
    Plot ROC and PR curves for both logistic regression and k-NN side by side.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba_log : np.ndarray or pd.Series
        Predicted probabilities from logistic regression
    y_pred_proba_knn : np.ndarray or pd.Series
        Predicted probabilities from k-NN
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2 subplots (ROC and PR curves)
    """
    # TODO: Implement comparison plotting
    # - Create figure with 1x2 subplots
    figure, axes = plt.subplots(1, 2)
    # - Left: ROC curves for both models
    auroc_log = generate_auroc_curve(y_true, y_pred_proba_log, "Log Regression", None, ax=axes[0])
    auroc_kNN = generate_auroc_curve(y_true, y_pred_proba_knn, "kNN", None, ax=axes[0])
    # - Right: PR curves for both models
    auprc_log = generate_auprc_curve(y_true, y_pred_proba_log, "Log Regression", None, ax=axes[1])
    auprc_kNN = generate_auprc_curve(y_true, y_pred_proba_knn, "kNN", None, ax=axes[1])
    # - Add legends with AUROC/AUPRC scores
    # - Save to output_path if provided
    if output_path != None:
        figure.savefig(output_path)
    # - Return figure
    return figure
