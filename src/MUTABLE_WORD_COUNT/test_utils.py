'''Import tested functions'''
import utils

def test_word_count():
    '''
    Test cases
    '''
    batch = ['Hi', 'I am', 'Peter'] # a batch of texts
    count = utils.word_count(batch=batch)
    assert count == {'Hi': 1, 'I': 1, 'am': 1, 'Peter': 1}


def test_word_count_tricky():
    '''
    Test tricky mutability cases
    '''
    # two batches with some similar words
    batch_1 = ['Hello', 'my name is', 'George']
    batch_2 = ['Hello', 'my name is', 'Lisa']

    count_1 = utils.word_count(batch=batch_1)
    count_2 = utils.word_count(batch=batch_2)

    assert count_1 == {'Hello': 1, 'my': 1, 'name': 1, 'is': 1, 'George': 1} and\
           count_2 == {'Hello': 1, 'my': 1, 'name': 1, 'is': 1, 'Lisa': 1}
