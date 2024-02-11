"""MFT negation testing"""
from typing import List
from typing import Tuple

import spacy
from checklist.expect import Expect
from checklist.perturb import Perturb
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT
from model import SentimentModel

# Initialize spaCy language processing pipeline with the English model
nlp = spacy.load("en_core_web_sm")

def handle_negation(sentence: str) -> str:
    """
    Add (if there is not) or remove (if there is)
    negation for a sentence
    """
    doc = nlp(sentence)
    if Perturb.add_negation(doc) is None:
        return Perturb.remove_negation(doc)
    return Perturb.add_negation(doc)

def run_negation_test(model: SentimentModel, sents: List[str]) -> MFT:
    """Negation NLP task testing using MFT"""
    # Wrap the model for checklist
    wrapped_model = PredictorWrapper.wrap_softmax(model.predict_proba)

    # Either add or remove negation from the sentence
    testcases = [[sent, handle_negation(sent)] for sent in sents]

    # Define the response function
    # Check documentation for more details
    # Return [True, True] if the testcase is passed
    # Return [False, False] if the testcase is failed
    def response(xs, preds, confs, labels=None, meta=None):
        """Subfunction to determinte if MFT is passed or failed"""
        diff = abs(confs[0][2] - confs[1][2])
        threshold = 0.3
        return [diff >= threshold, diff >= threshold]

    # Create and run the test
    # Expect should work with testcases
    test = MFT(data=testcases, expect=Expect.testcase(response))
    test.run(wrapped_model)

    return test


if __name__ == "__main__":
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

    test = run_negation_test(model, sents)

    def format_example(x, pred, conf, label=None, meta=None):
        return f"{x} (pos. class conf.: {conf[2]:.2f})"

    print(test.summary(n=5, format_example_fn=format_example))
