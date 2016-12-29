# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def text_to_unigrams(text):
    return list(text)

TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data("train")]
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("dev")]
TEST   = [(l,text_to_bigrams(t)) for l,t in read_data("test")]

TRAIN_UNIGRAM = [(l,text_to_unigrams(t)) for l,t in read_data("train")]
DEV_UNIGRAM   = [(l,text_to_unigrams(t)) for l,t in read_data("dev")]

from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

fc_unigram = Counter()
for l,feats in TRAIN_UNIGRAM:
    fc_unigram.update(feats)

# use 100 common letters as vocabulary
unigrams_voc = set([x for x,c in fc_unigram.most_common(100)])

# unigrams_voc = set()
# for l,feats in TRAIN_UNIGRAM:
#     unigrams_voc = unigrams_voc | set(feats)



F2I_UNI = {f:i for i,f in enumerate(list(sorted(unigrams_voc)))} 

# 600 most common bigrams in the training set.
vocab = set([x for x,c in fc.most_common(600)])

# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}

# IDs to label strings
I2L = {i:l for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}

# feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

