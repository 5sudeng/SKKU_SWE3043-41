from torchvision import datasets
from base import BaseDataLoader
from data.transformation.preprocessing import MyDataset


class LSTMDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader -> 수정해서 사용
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.2, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = MyDataset(data_dir, 3)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
