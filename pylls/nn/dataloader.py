from typing import Any 
import numpy as np

class Dataloader:
    """
    A generic DataLoader for batching datasets of inputs and targets.
    
    Attributes:
        data (np.ndarray): The input data (features).
        targets (np.ndarray): The target labels.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at the start of each epoch.
    """
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, batch_size: int = 64, shuffle: bool = True):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        # Shuffle the indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0  # Initialize the starting index for iteration
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        # Batch slicing
        start = self.current_index
        end = min(start + self.batch_size, self.num_samples)
        batch_indices = self.indices[start:end]
        
        # Advance current index
        self.current_index = end
        
        return self.data[batch_indices], self.targets[batch_indices]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
