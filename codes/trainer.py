import os

from flair.datasets import DataLoader
from sklearn.model_selection import train_test_split
from torcheeg.model_selection import train_test_split_per_subject_cross_trial
import numpy as np
from model import Net
import torch
import torch.nn as nn
from dataset import EEG_dataset
from draw import logger_configure, plot_confusion_matrix
import pandas as pd
from copy import copy
class Trainer():
    def __init__(self, dataset):
        self.dataset = dataset
        self.info = dataset.info
        self.model = None
        self.epochs = 10
        self.batch_size = 64
        self.logger = logger_configure()
        # Create the model
        self.model = Net(num_classes=4, input_channels=22)
        # Create the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    def cross_session(self):
        """
        Cross-session training

        :return: None
        """
        all_subjects = list(set(self.info['subject_id']))
        for sub in all_subjects:
            train_dataset, test_dataset = train_test_split_per_subject_cross_trial(self.dataset, subject=sub, split_path=r'.torcheeg/model_selection')
            self.__train(train_dataset, sub)
            # Test the model
            test_acc = self.__test(test_dataset, sub)
            self.logger.info(f'Subject: {sub}, Test Accuracy: {test_acc}')
            # Save the model
            # torch.save(self.model.state_dict(), f"model_sub_{sub}.pth")


    def __train(self, train_dataset, sub):
        """
        Train the model
        :param x_train: training data
        :param y_train: training label
        :return: None
        """
        # Create the data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # Set the model to training mode
        self.model.train()
        val_max_acc = 0

        for epoch in range(self.epochs):
            for i, (data, label) in enumerate(train_loader):
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                output = self.model(data)
                # Compute the loss
                loss = self.criterion(output, label)
                # Backward pass
                loss.backward()
                # Update the weights
                self.optimizer.step()
            # Print the loss
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}")
        os.makedirs(fr'results/models', exist_ok=True)
        torch.save(self.model.state_dict(), 'results/models/model_{sub}.pth')
        print("Model saved")

    def __validate(self, val_dataset):
        """
        Validate the model
        :param x_test: testing data
        :param y_test: testing label
        :return: None
        """
        # Create the data loader
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        # Set the model to evaluation mode
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in val_loader:
                # Forward pass
                output = self.model(data)
                # Compute the loss
                loss = self.criterion(output, label)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc}%")
        return val_acc

    def __test(self, test_dataset, sub):
        """
        Test the model
        :param x_test: testing data
        :param y_test: testing label
        :return: None
        """
        # Create the data loader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        # Set the model to evaluation mode
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            preds = []
            labels = []
            for data, label in test_loader:
                # Forward pass
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                preds.append(predicted.cpu().numpy())
                labels.append(label.cpu().numpy())

        preds = np.concatenate(preds, axis=0).squeeze()
        labels = np.concatenate(labels, axis=0).squeeze()
        # Plot the confusion matrix
        plot_confusion_matrix(sub, preds, labels)
        test_acc = 100 * correct / total
        print(f"Test Accuracy: {test_acc}%")
        return test_acc