"""Reviews data loader"""
from typing import List, Generator, Tuple
from dataclasses import dataclass
import itertools

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

import torch
from transformers import PreTrainedTokenizer

@dataclass
class DataLoader:
    """Client reviews loader class"""
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: str = None # None, "max_length", or "batch"

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        # open the file and count the number of lines
        total_lines = 0
        with open(self.path, 'r', encoding='utf-8') as file:
            next(file) # skip the header
            for _ in file:
                total_lines += 1

        # calculate the number of batches
        num_batches = total_lines // self.batch_size
        if total_lines % self.batch_size != 0:
            num_batches += 1

        return num_batches

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        tokenized_texts = [self.tokenizer.encode(text,
                                                 add_special_tokens=True,
                                                 max_length=self.max_length)
                           for text in batch]
        return tokenized_texts

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        texts = []
        labels = []

        with open(self.path, 'r', encoding='utf-8') as file:
            # skip the header line
            next(file)

            # calculate the starting line number for the batch
            start_line = self.batch_size * i

            # read and process the batch
            for line_num, line in enumerate(file):
                # skipping data before the batch
                if line_num < start_line:
                    continue
                # if no data further, break the loop
                if line is None:
                    break

                # split the line into text and label (assuming comma-separated values)
                _, _, _, label, text = line.strip().split(',', 4)
                texts.append(text)
                # encoding the sentiment label
                if label == 'positive':
                    label = 1
                elif label == 'negative':
                    label = -1
                else:
                    label = 0
                labels.append(label)

                # break the loop if the desired batch size is reached
                if len(texts) == self.batch_size:
                    break

        return texts, labels

    def _padded(self, tokens: List[List[int]]) -> List[List[int]]:
        """Pad the token vocab-indices with zeros"""
        pad_token = 0
        padded_tokens = []

        # if no padding type specified, return the input
        if self.padding is None:
            return tokens

        # static padding (independent on the batch size; fixed maxlen)
        if self.padding == 'max_length':
            for token_ids in tokens:
                # if a tokenized text is too short, supplement it with 0's
                if self.max_length > len(token_ids):
                    token_ids.extend([pad_token] * (self.max_length - len(token_ids)))
            return padded_tokens

        # dynamic padding (padding amount varies for each batch)
        if self.padding == 'batch':
            padded_tokens = list(zip(*itertools.zip_longest(*tokens, fillvalue=pad_token)))
            return padded_tokens

        raise ValueError('Wrong value for "padding" parameter!')

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        # loading and tokenizing
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        # padding
        tokens = self._padded(tokens)
        return tokens, labels


def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    """Masking significant and empty tokens away from each other"""
    # significant values have weight = 1, empty - 0
    mask = [list(map(lambda el: 1 if el != 0 else 0, row)) for row in padded]
    return mask

def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    # attention mask
    mask = attention_mask(tokens)

    # calculate embeddings
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)

    # no grad calculations
    with torch.no_grad():
        last_hidden_states = model(tokens, attention_mask=mask)

    # embeddings for [CLS]-tokens
    features = last_hidden_states[0][:,0,:].tolist()
    return features

def evaluate(model, embeddings, labels, cv=5) -> List[float]:
    """Evaluate prediction model quality having embeddings layer as input"""
    # model scores for each fold
    fold_scores = []
    # creating the KFold validation
    kf = KFold(n_splits=cv)
    # KFold cross validation predictions for each fold
    for train_index, test_index in kf.split(embeddings):
        # fitting the model on the train set
        # and predicting on the test set
        model.fit(embeddings[train_index], labels[train_index])
        y_pred = model.predict_proba(embeddings[test_index])

        # calculating the score on the test set
        score = log_loss(labels[test_index], y_pred)
        fold_scores.append(score)

    return fold_scores
