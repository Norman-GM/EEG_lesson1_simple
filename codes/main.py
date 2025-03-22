from preprocessing import load_data
from preprocessing import download_data
from trainer import Trainer
def main():
    # download_data(r'D:\dataset')
    dataset = load_data(r'D:\dataset\BCICIV2a\MNE-bnci-data\database\data-sets\001-2014')
    trainer = Trainer(dataset)
    trainer.cross_session()



if __name__ == "__main__":
    main()