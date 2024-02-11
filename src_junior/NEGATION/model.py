from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class SentimentModel:
    def __init__(
        self,
        name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name)

    def predict_proba(self, inputs: List[str]) -> np.ndarray:
        tokens = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = self.model(**tokens).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.detach().numpy()

if __name__ == '__main__':
    model = SentimentModel()
    sents = [
        "The delivery was swift and on time.",
        "I wasn't disappointed with the service.",
        "The food arrived cold and unappetizing.",
        "Their app is quite user-friendly and intuitive.",
        "I didn't find their selection lacking.",
        "The delivery person was rude and impatient.",
        "They always have great deals and offers.",
        "I haven't had any bad experiences yet.",
        "I was amazed by the quick response to my complaint.",
        "Their tracking system isn't always accurate.",
    ]
    print(model.predict_proba(sents))