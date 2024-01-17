"""Data processing"""
from typing import List

import numpy as np

class DataProcessor:
    """Data processing class"""
    def __init__(self, data):
        self.data: List[float] = data[:]
        self.processed_data_: List[float] = None

    def process(self) -> None:
        """Processes input numerical data"""
        data_np = np.array(self.data)
        mean = np.array(self.data).mean()
        # mean shifting
        self.processed_data_: np.ndarray = data_np - mean
        self.processed_data_ = self.processed_data_.tolist()

    def save_to_file(self, filename: str) -> None:
        """Saves data values in a text file line-by-line"""
        # if data haven't been processed, do nothing
        if self.processed_data_ is None:
            return

        # writing data elements line-by-line
        with open(filename, 'w', encoding='utf-8') as f:
            for line in self.processed_data_:
                f.write(f"{line}\n")
