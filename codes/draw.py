import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
def logger_configure(log_file=None):
    """
    Configure the logger
    """
    # Create a
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create a file handler
    if log_file:
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


def plot_confusion_matrix(sub, preds, labels):
    """
    Plot confusion matrix
    :param sub: subject number
    :param preds: predicted labels
    :param labels: true labels
    :return: None
    """

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Plot confusion matrix
    classes = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for Subject {sub}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    os.makedirs(r'results/figures/confusion_matrix', exist_ok=True)
    plt.savefig(fr'results/figures/confusion_matrix/cm_{sub}.png')
    plt.close()



