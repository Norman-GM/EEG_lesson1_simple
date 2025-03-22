from preprocessing import load_data, download_data
from trainer import Trainer
from utils import seed_everything

def main():
    """
    Main entry point for the EEG classification project.
    
    This function:
    1. Sets random seeds for reproducibility
    2. Downloads the dataset if not present
    3. Loads and preprocesses the EEG data
    4. Initializes the trainer
    5. Performs cross-session training and evaluation
    """
    # Set random seeds for reproducibility
    seed_everything(2025)
    
    # Download the BNCI2014_001 dataset if not already present
    # download_data(r'D:\dataset')
    
    # Load and preprocess the dataset
    dataset = load_data(r'D:\dataset\BCICIV2a\MNE-bnci-data\database\data-sets\001-2014')
    
    # Initialize the trainer with the dataset
    trainer = Trainer(dataset)
    
    # Perform cross-session training and evaluation
    trainer.cross_session()


if __name__ == "__main__":
    main()
