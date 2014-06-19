import numpy as np


def load_mikolov(vocab):
    """load Mikolov's word embeddings"""

    result = np.zeros((len(vocab), 300), float)

    with open("lexicon.CI.txt") as f:
        for line in f.readlines():
            parts = line.split(' ')
            weights = np.array(parts[1:]).astype(float)
            result[vocab[parts[0]], :] = weights

    with open("lexicon.CS.txt") as f:
        for line in f.readlines():
            parts = line.split(' ')
            weights = np.array(parts[1:]).astype(float)
            result[vocab[parts[0]], :] = weights

    # useless
    #with open("lexicon.unk.txt") as f:
    #    for line in f.readlines():
    #        parts = line.split('\n')
    #        weights = np.zeros((300), dtype=float)
    #        result[vocab[parts[0]], :] = weights

    return result
