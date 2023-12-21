from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, X_train, y_train):
        self.n_data = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __len__(self):
        return self.n_data
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]