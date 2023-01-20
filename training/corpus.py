import nltk.corpus
import random
import re
import more_itertools
from typing import Callable

re_word = re.compile(r'^[a-zA-Z0-9].*$')

def word_join(words: list[str]) -> str:
    word_spaces = []
    for w in words:
        if len(word_spaces) > 0 and re_word.match(w):
            word_spaces.append(' ')
        word_spaces.append(w)
    return ''.join(word_spaces)

# Returns a list of batches
# Each batch is of length batch_size and is a list of windows
# Each window is of length window_size and is a list of tokens (integers)
def get_corpus(seed:int, n:int, train:bool, batch_size:int, window_size:int, tokenizer:Callable) -> list[list[list[int]]]:
    c = nltk.corpus.brown
    s = c.sents()
    cutoff = int(len(s) * 0.8)
    if train:
        s = s[:cutoff]
    else:
        s = s[cutoff:]
    random.seed(seed)
    randoms = random.sample(range(len(s)), k=n)
    
    result = []
    for r in randoms:
        tokens = []
        i = 0
        while len(tokens) < window_size:
            sentence = word_join(s[r + i % len(s)])
            tokens += tokenizer.encode(sentence)
            i += 1
        tokens = tokens[:window_size]
        result.append(tokens)
    
    return more_itertools.batched(result, batch_size)

