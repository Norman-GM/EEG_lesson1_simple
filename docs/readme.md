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
- `dataset.py`: Dataset class for handling EEG data

## Features

- **Data Processing**: Utilizing TorchEEG for efficient EEG data handling
- **Neural Network Architecture**: 
  - Spatial filtering layer to extract patterns across EEG channels
  - Temporal filtering layer to capture time-based features
  - Fully connected layers for classification
- **Cross-Session Validation**: Train and test on different recording sessions for each subject
- **Performance Visualization**: Confusion matrices to analyze classification performance
- **Reproducibility**: Seed setting for consistent results across runs
- **Model Saving**: Trained models are saved for later use or analysis

## Requirements

- Python 3.x
- PyTorch
- TorchEEG
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- MOABB (Mother of all BCI Benchmarks)

## Installation

```bash
# Create a conda environment
conda create -n eeg_bci python=3.8
conda activate eeg_bci

# Install required packages
pip install torch torchvision torchaudio
pip install torcheeg
pip install numpy pandas matplotlib scikit-learn
pip install moabb flair
```

## Usage

1. Set up the dataset path in `main.py`
2. Run the script to train and evaluate the model:

```bash
python codes/main.py
```

## Model Architecture

The model uses a two-layer convolutional neural network:
- **First layer**: Spatial filtering across EEG channels (captures spatial patterns)
- **Second layer**: Temporal filtering across time samples (captures temporal features)
- **Fully connected layers**: For dimensionality reduction and classification

## Results

The model produces confusion matrices for each subject showing classification performance across the four motor imagery classes. Results are saved to the `results/figures/confusion_matrix` directory.

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset: BNCI Horizon 2020 (BCI Competition IV 2a)
- TorchEEG framework for EEG data processing
