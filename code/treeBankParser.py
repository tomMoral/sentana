import sys
from numpy.random import randint


def write_dataset(rawInputFile, outputDir):
    data = open(rawInputFile, 'r')
    out = open(outputDir + "/datasetSentences.txt", 'w')
    out.write("sentence_index\tsentence\n")
    for i, line in enumerate(data):
        out.write("%i\t" % i + line)


def convert_all(outputDir, trees, split=None):
    out_raw = open(outputDir + "/datasetSentences.txt", 'w')
    out_raw.write("sentence_index\tsentence\n")

    out_S = open(outputDir + "/STree.txt", 'w')
    out_SOS = open(outputDir + "/SOStr.txt", 'w')
    out_content = open(outputDir + "/TreeIDs.txt", 'w')

    out_label = open(outputDir + "/sentiment_labels.txt", 'w')
    out_label.write("phrase ids|sentiment values\n")

    out_split = open(outputDir + "/datasetSplit.txt", 'w')
    out_split.write("sentence_index,splitset_label\n")

    out_dico = open(outputDir + "/dictionary.txt", 'w')

    i = 0   # count part of sentences
    i_line = 0   # count sentences

    def rand(split):
        if split is None:
            return max(1, randint(6) - 2)
        else:
            return split

    for line in trees:
        t = read_tree(line)
        t.print_parent(out_S)
        t.print_content(out_SOS, sep='|')
        out_raw.write('%i\t' % i_line)
        t.print_content(out_raw, sep=' ')
        out_split.write('%i,%i\n' % (i_line, rand(split)))
        content = ""
        for e in t.traverse():
            out_dico.write(e.get_content() + '|%i\n' % i)
            if e.label >= 0 and e.label <= 1:
                out_label.write('%i|%4f\n' % (i, e.label))
            content += '%i|' % i
            i += 1
        out_content.write(content[:-1] + '\n')
        i_line += 1


def convert_all_from(outputDir, *filename):
    convert_all(outputDir, read_all(filename))


def read_all(filenames):
    for f in filenames:
        print f
        source = open(f)
        for line in source:
            yield line


def read_tree(tree_string):
    segment = tree_string.split("(")[1:]

    def read_segment(i):
        if segment[i][-2] == ')':
            s = segment[i].split(" ", 2)
            label = int(s[0])
            content = s[1].split(")", 2)[0]
            return (Node(label=label, content=content), i + 1)
        else:
            label = int(segment[i])
            (left, j) = read_segment(i + 1)
            (right, j) = read_segment(j)
            return (Node(label=label, left=left, right=right), j)
    (t, _) = read_segment(0)
    t.set_index()
    t.set_parent()
    return t


class Node(object):
    def __init__(self, left=None, right=None, label=None, content=None):
        self.index = None
        self.left = left
        self.right = right
        if label == 3:
            self.label = 0.99
        elif label == 1:
            self.label = 0.01
        else:
            self.label = -1
        self.content = content
        self.parent = None

    def is_leaf(self):
        return self.left is None or self.right is None

    def get_content(self, sep=' '):
        if self.is_leaf():
            return self.content
        else:
            return self.left.get_content(sep) + sep \
                + self.right.get_content(sep)

    def print_content(self, output, sep=' '):
        output.write(self.get_content(sep) + '\n')

    def set_content(self):
        if not self.is_leaf():
            self.content = self.left.set_content() + ' ' \
                + self.right.set_content()
        return self.content

    def print_parent(self, output):
        parents = ""
        for n in self.traverse():
            if n.parent is not None:
                parents += "%i|" % n.parent
        output.write(parents[:-1] + '\n')

    def set_index(self):
        i = 1
        for n in self.traverse():
            n.index = i
            i += 1

    def set_parent(self, i=0):
        self.parent = i
        if not self.is_leaf():
            self.left.set_parent(self.index)
            self.right.set_parent(self.index)

    def traverse_leaf(self):
        if self.is_leaf():
            yield self
        else:
            for e in self.left.traverse_leaf():
                yield e
            for e in self.right.traverse_leaf():
                yield e

    def traverse(self):
        for e in self.traverse_leaf():
            yield e

        def traverse_rec(tree):
            if not tree.is_leaf():
                for e in traverse_rec(tree.left):
                    yield e
                for e in traverse_rec(tree.right):
                    yield e
                yield tree

        for e in traverse_rec(self):
            yield e


def test():
    t = read_tree("(3 (-1 bon) (-1 (-1 (-1 goût) (-1 (-1 ,) (-1 facile))) (-1 (-1 à) (-1 préparer))))")
    t.print_content(sys.stdout)
    print
    t.print_parent(sys.stdout)
    print
    # t2 = read_tree("(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's)) (2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 '')) (2 and)) (3 (2 that) (3 (2 he) (3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) (2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) (2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .)))")

    #t2 = read_tree("(0 (2 A) (0 (3 real) (1 (0 snooze) (2 .))))")
    #expected result: 7|6|5|5|6|7|0 -> ok

    t2 = read_tree("(3 (2 I) (3 (3 (4 love) (2 (2 (2 (2 the) (2 (2 opening) (2 scenes))) (2 (2 of) (2 (2 a) (2 (2 wintry) (2 (3 New) (2 (2 York) (2 City))))))) (2 (2 in) (2 1899)))) (2 .)))")
    #expected result : 27|25|22|21|21|20|19|18|17|16|16|15|15|26|
                                #  24|17|18|19|20|23|22|23|24|25|26|27|0
    #got :             27|25|16|15|15|21|20|19|18|17|17|23|23|26|
                                #  16|22|18|19|20|21|22|24|24|25|26|27|0|
    t2.print_content(sys.stdout)
    t2.print_parent(sys.stdout)


if __name__ == '__main__':
    # test()
    convert_all_from("../data/ciao2",
                     "../data/ciao/devCiao.txt",
                     "../data/ciao/trainCiao.txt",
                     "../data/ciao/testCiao.txt")
