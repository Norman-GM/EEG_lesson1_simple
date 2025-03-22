import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List, Optional, Union, Any

def logger_configure(log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger for tracking the training process.
    
    Sets up both console and file logging if a log file path is provided.
    
    Args:
        log_file: Path to the log file (optional)
        
    Returns:
        Configured logger instance
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def plot_confusion_matrix(sub: int, preds: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot and save a confusion matrix for classification results.
    
    Creates a normalized confusion matrix showing the prediction accuracy
    for each class.
    
    Args:
        sub: Subject ID for naming the output file
        preds: Array of model predictions
        labels: Array of true labels
    """
    # Define class names for motor imagery tasks
    classes = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Normalize the confusion matrix (rows sum to 1)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure for the confusion matrix
    plt.figure(figsize=(8, 6))
    
    # Plot the confusion matrix as a heatmap
    sns.heatmap(
        cm, 
        annot=True,           # Show the values in each cell
        fmt='.2f',            # Format as 2 decimal points
        cmap='Blues',         # Use Blues colormap
        xticklabels=classes,  # Label x-axis with class names
        yticklabels=classes   # Label y-axis with class names
    )
    
    # Add title and axis labels
    plt.title(f'Confusion Matrix for Subject {sub}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Create directory for saving the figure if it doesn't exist
    os.makedirs(r'results/figures/confusion_matrix', exist_ok=True)
    
    # Save the figure
    plt.savefig(fr'results/figures/confusion_matrix/cm_{sub}.png')
    
    # Close the figure to free memory
    plt.close()
