'''WORD COUNT'''

def word_count(batch, count=None):
    '''
    Calculates a counter dictionary for a batch's texts words
    (altered avoiding mutable default parameters)
    '''
    if count is None:
        count = {}
    for text in batch:
        for word in text.split():
            count[word] = count.get(word, 0) + 1
    return count
