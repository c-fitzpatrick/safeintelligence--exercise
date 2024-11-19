import time
import torch
from torch.utils.data import Dataset
import pandas as pd

from concurrent.futures import ProcessPoolExecutor

class SingleProcessDataset(Dataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using single process...")
        
        self.data = pd.read_csv(csv_file)
        self.features = torch.FloatTensor(self.data[['x1', 'x2', 'x3']].values)
        self.labels = torch.LongTensor(self.data['label'].values)
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultiProcessDataset(SingleProcessDataset):
    def __init__(self, csv_file, chunksize=1000):
        start_time = time.time()
        print("Loading data using multi process...")

        ########### YOUR CODE HERE ############

        chunks = pd.read_csv(csv_file, chunksize=chunksize, iterator=True)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.return_chunk, chunk) for chunk in chunks]
            self.data = pd.concat([future.result() for future in futures])

        self.features = torch.FloatTensor(self.data[['x1', 'x2', 'x3']].values)
        self.labels = torch.LongTensor(self.data['label'].values)
        
        ########### END YOUR CODE  ############
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    @staticmethod
    def return_chunk(chunk):
        return chunk
