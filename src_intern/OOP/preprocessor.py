"""Preprocessing abstact class"""
from abc import ABC, abstractmethod
from typing import Dict

class DataPreprocessor(ABC):
    """Data preprocessor base class"""
    @abstractmethod
    def preprocess(self, data):
        """Preprocesses numerical data"""
        return

class OutlierRemover(DataPreprocessor):
    """Class removing outliers"""
    def preprocess(self, data):
        """Removes outliers"""
        prep_data = [num for num in data if num <= 10]
        return prep_data

class Normalizer(DataPreprocessor):
    """Class normalizing data"""
    def preprocess(self, data):
        """Normalizes data"""
        prep_data = [num/10 for num in data]
        return prep_data

class Encoder(DataPreprocessor):
    """Class encoding data"""
    def __init__(self, encoding_dict: Dict[str, int]):
        self.encoding_dict = encoding_dict

    def preprocess(self, data):
        """Encodes data"""
        prep_data = [self.encoding_dict[el] for el in data]
        return prep_data
