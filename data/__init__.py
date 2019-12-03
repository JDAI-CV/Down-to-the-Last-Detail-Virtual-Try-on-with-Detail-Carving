import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.regular_dataset import RegularDataset

def create_dataset(opt, augment, istrain, uniform_val):

    data_loader = CustomDatasetDataLoader(opt, augment, istrain, uniform_val)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, augment, istrain, uniform_val):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = RegularDataset(opt, augment, istrain, uniform_val)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
        

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    # def __iter__(self):
    #     """Return a batch of data"""
    #     for i, data in enumerate(self.dataloader):
    #         if i * self.opt.batch_size >= self.opt.max_dataset_size:
    #             break
    #         yield data
