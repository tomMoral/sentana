from os import path

DATASET = '../data'
if __name__ == '__main__':
    with open(path.join(DATASET, 'STree.txt')) as f:
        tree1 = f.readline()
    with open(path.join(DATASET, 'datasetSentences.txt')) as f:
        sent1 = f.readline()
        sentences = []
        for line in f.readlines():
            sent = line.split('\t')[1]
            sent = sent.replace('\n', '')
            sentences.append(sent)

    print len(sentences)
