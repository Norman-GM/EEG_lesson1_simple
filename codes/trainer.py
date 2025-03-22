import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

from flair.datasets import DataLoader
from sklearn.model_selection import train_test_split
from torcheeg.model_selection import train_test_split_per_subject_cross_trial

from model import Net
from dataset import EEG_dataset
from draw import logger_configure, plot_confusion_matrix

class Trainer:
    """
    A trainer class for EEG classification models.
    
    This class handles training, validation, and testing of EEG classification models
    using cross-session validation approaches.
    
    Attributes:
        dataset: The EEG dataset used for training and testing
        info: Metadata about the dataset
        model: The neural network model
        epochs: Number of training epochs
        batch_size: Batch size for training
        logger: Logging utility for tracking progress
        criterion: Loss function
        optimizer: Optimization algorithm
    """
    def __init__(self, dataset: EEG_dataset):
        """
        Initialize the Trainer with dataset and default configuration.
        
        Args:
            dataset: The EEG dataset used for training and testing
        """
        self.dataset = dataset
        self.info = dataset.info
        self.model = None
        self.epochs = 10
        self.batch_size = 64
        self.logger = logger_configure()
        
        # Create the model (4 classes for motor imagery: left hand, right hand, feet, tongue)
        self.model = Net(num_classes=4, input_channels=22)  # 22 EEG channels
        
        # Create the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def cross_session(self) -> None:
        """
        Perform cross-session training and evaluation.
        
        This method trains and tests the model for each subject separately,
        using a leave-one-session-out approach for each subject.
        """
        # Get list of unique subjects
        all_subjects = list(set(self.info['subject_id']))
        
        for sub in all_subjects:
            self.logger.info(f"Processing subject {sub}")
            
            # Split data for this subject into train and test sets
            train_dataset, test_dataset = train_test_split_per_subject_cross_trial(
                self.dataset, 
                subject=sub, 
                split_path=r'.torcheeg/model_selection'
            )
            
            # Train the model
            self.__train(train_dataset, sub)
            
            # Test the model
            test_acc = self.__test(test_dataset, sub)
            self.logger.info(f'Subject: {sub}, Test Accuracy: {test_acc:.2f}%')

    def __train(self, train_dataset: Any, sub: int) -> None:
        """
        Train the model on the given dataset.
        
        Args:
            train_dataset: Dataset to train on
            sub: Subject ID for model naming
        """
        # Create the data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Set the model to training mode
        self.model.train()
        val_max_acc = 0

        # Training loop
        for epoch in range(self.epochs):
            running_loss = 0.0
            batch_count = 0
            
            for i, (data, label) in enumerate(train_loader):
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Compute the loss
                loss = self.criterion(output, label)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                batch_count += 1
            
            # Print epoch statistics
            avg_loss = running_loss / batch_count
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Save the trained model
        os.makedirs(fr'results/models', exist_ok=True)
        torch.save(self.model.state_dict(), f'results/models/model_{sub}.pth')
        print(f"Model for subject {sub} saved")

    def __validate(self, val_dataset: Any) -> float:
        """
        Validate the model on the given dataset.
        
        Args:
            val_dataset: Dataset for validation
            
        Returns:
            Validation accuracy as a percentage
        """
        # Create the data loader
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Set the model to evaluation mode
        self.model.eval()
        correct = 0
        total = 0
        
        # No gradients needed for validation
        with torch.no_grad():
            for data, label in val_loader:
                # Forward pass
                output = self.model(data)
                
                # Get predictions
                _, predicted = torch.max(output.data, 1)
                
                # Count correct predictions
                total += label.size(0)
                correct += (predicted == label).sum().item()
        
        # Calculate accuracy
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        return val_acc

    def __test(self, test_dataset: Any, sub: int) -> float:
        """
        Test the model on the given dataset and visualize results.
        
        Args:
            test_dataset: Dataset for testing
            sub: Subject ID for result naming
            
        Returns:
            Test accuracy as a percentage
        """
        # Create the data loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Set the model to evaluation mode
        self.model.eval()
        correct = 0
        total = 0
        
        # No gradients needed for testing
        with torch.no_grad():
            preds = []
            labels = []
            
            for data, label in test_loader:
                # Forward pass
                output = self.model(data)
                
                # Get predictions
                _, predicted = torch.max(output.data, 1)
                
                # Count correct predictions
                total += label.size(0)
                correct += (predicted == label).sum().item()
                
                # Store predictions and true labels for confusion matrix
                preds.append(predicted.cpu().numpy())
                labels.append(label.cpu().numpy())

        # Concatenate all predictions and labels
        preds = np.concatenate(preds, axis=0).squeeze()
        labels = np.concatenate(labels, axis=0).squeeze()
        
        # Visualize results with confusion matrix
        plot_confusion_matrix(sub, preds, labels)
        
        # Calculate and return accuracy
        test_acc = 100 * correct / total
        print(f"Test Accuracy for subject {sub}: {test_acc:.2f}%")
        return test_acc
