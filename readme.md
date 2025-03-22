# EEG BCI Classification Project

This project implements a neural network-based approach for classifying EEG data from the BNCI2014_001 (BCI Competition IV 2a) dataset. The implementation uses PyTorch and TorchEEG for processing and classifying motor imagery EEG signals.

## Overview

The project classifies four motor imagery tasks (left hand, right hand, feet, tongue movements) from EEG data using a convolutional neural network. It implements a cross-session validation approach to evaluate model performance.

## Project Structure

- `main.py`: Entry point that loads data and initiates training
- `preprocessing.py`: Functions for data downloading and loading
- `model.py`: Neural network architecture definition
- `trainer.py`: Training and evaluation logic
- `draw.py`: Visualization utilities for results
- `utils.py`: Helper functions including random seed setting

## Features

- Data processing with TorchEEG
- Convolutional neural network architecture for EEG classification
- Cross-session training and evaluation
- Confusion matrix visualization for performance analysis
- Support for hyperparameter tuning
- Model saving and evaluation metrics

## Requirements

- Python 3.x
- PyTorch
- TorchEEG
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- MOABB (Mother of all BCI Benchmarks)

## Usage

1. Set up the dataset path in `main.py`
2. Run the script to train and evaluate the model:

```bash
python codes/main.py
```

## Model Architecture

The model uses a two-layer convolutional neural network:
- First layer: Spatial filtering across EEG channels
- Second layer: Temporal filtering across time samples
- Fully connected layers for classification

## Results

The model produces confusion matrices for each subject showing classification performance across the four motor imagery classes. Results are saved to the `results/figures/confusion_matrix` directory.

## License

This project is for educational and research purposes.