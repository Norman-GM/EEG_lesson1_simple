from sklearn.preprocessing import StandardScaler
class EEG_dataset():
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label
