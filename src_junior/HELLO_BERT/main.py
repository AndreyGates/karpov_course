"""Applying the data loader"""
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.linear_model import LogisticRegression

import numpy as np
from utils import DataLoader, review_embedding, evaluate

MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
bert = DistilBertModel.from_pretrained(MODEL_NAME)

loader = DataLoader('src_JUNIOR/HELLO_BERT/reviews.csv', tokenizer, max_length=128, padding='batch')

X = []
y = []
for tokens, labels in loader:
    features = review_embedding(tokens, bert)
    X.extend(features)
    y.extend(labels)

model = LogisticRegression()
scores = evaluate(model, np.array(X), np.array(y))
print(scores)
