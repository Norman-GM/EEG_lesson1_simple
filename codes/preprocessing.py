from torcheeg.datasets import BCICIV2aDataset
from torcheeg import transforms
from moabb.datasets import BNCI2014_001
def download_data(dir):
    """
    Download the data from the given directory
    :param dir: directory to save the data
    :return: None
    """
    # Download the data using moabb

    dataset = BNCI2014_001()
    dataset.download(subject_list=list(range(1, 10)), path=dir)


def load_data(dir):
    # Load the data using torcheeg
    dataset = BCICIV2aDataset(root_path=dir, chunk_size=250,
                          online_transform=transforms.Compose([
                              transforms.To2d(),
                              transforms.ToTensor()
                          ]),
                          label_transform=transforms.Compose([
                              transforms.Select('label'),
                              transforms.Lambda(lambda x: x - 1)
                          ]), io_path=r'.torcheeg/preprocess')
    return dataset