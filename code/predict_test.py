import random
import utils
import numpy as np



def get_frequencies(bigram_list, indexed_vocabulary):
    ''' Gets a bigram list and an indexed vocabulary and return a numpy vector of the frequencies of 
        each bigram in the given bigram_list. The frequencies are normalized by the length of the list.'''
    features = np.zeros(len(indexed_vocabulary))
    # place frequencies of the bigrams in the vocabulary within 'bigram_list' 
    for bi in set(bigram_list) & set(indexed_vocabulary.keys()):
        features[indexed_vocabulary[bi]] = bigram_list.count(bi)

    # return normalized frequencies
    return features / float(len(bigram_list))


def predict_test_to_file(model, params):
    # extract train and dev tests using utils
    test_set = utils.TEST

    # Convert train examples and dev examples to the format: (lang_id, frequencies) while 
    # frequencies is a vector that represents 600 common bigrams' normalized frequencies in
    # the example's text.
    test_data = [None] * len(test_set)
    for i, example in enumerate(test_set):
        cur_bigrams = example[1]
        test_data[i] = get_frequencies(cur_bigrams, utils.F2I)

    # create file and write the predictions
    wf = open('test_MLP.pred', 'w')
    for x in test_data:
        wf.write(utils.I2L[model.predict(x, params)] +'\n')

    wf.close()
    print 'finished writing predicted results to "test_MLP.pred"!!'

# if __name__ == '__main__':
#     vocabulary_size = len(utils.vocab)
#     languages_amount = len(utils.L2I)
#     zero_params = ll.create_classifier(vocabulary_size, languages_amount)
#     predict_test_to_file(zero_params)
    


