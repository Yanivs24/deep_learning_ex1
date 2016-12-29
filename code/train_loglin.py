import loglinear as ll
import random
import utils
import numpy as np

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return None

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        pred_label = ll.predict(features, params)
        if pred_label == label:
            good += 1 
        else:
            bad += 1

    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    # init variables
    consecutive_unimproves = 0
    best_params = list()
    best_dev_accuracy = -1
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features # numpy vector
            y = label    # a number
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss

            W, b = params
            # SGD update rules
            new_W = W - learning_rate * grads[0]
            new_b = b - learning_rate * grads[1]
            params = (new_W, new_b) # update params for next iteration

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy

        # save best params so far
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_params = params
            consecutive_unimproves = 0
        else:
            consecutive_unimproves += 1

        # If the performance on the dev set wasn't improved for 5 consecutive
        # epochs - finish training and return the best params so far
        if consecutive_unimproves == 5:
            break

    print 'Finished training - best accuracy on dev is: %s' % str(best_dev_accuracy)
    return best_params

def get_frequencies(word_list, indexed_vocabulary):
    ''' Gets a bigram list and an indexed vocabulary and return a numpy vector of the frequencies of 
        each bigram in the given word_list. The frequencies are normalized by the length of the list.'''
    features = np.zeros(len(indexed_vocabulary))
    # place frequencies of the bigrams in the vocabulary within 'word_list' 
    for bi in set(word_list) & set(indexed_vocabulary.keys()):
        features[indexed_vocabulary[bi]] = word_list.count(bi)

    # return normalized frequencies
    return  100 * features / float(len(word_list))

if __name__ == '__main__':
    
    models = ('unigram', 'bigram')
    model = 'bigram'

    # extract train and dev tests using utils
    if model == 'bigram':
        train_set = utils.TRAIN 
        dev_set = utils.DEV
        indexed_vocab = utils.F2I
    else:
        train_set = utils.TRAIN_UNIGRAM 
        dev_set = utils.DEV_UNIGRAM 
        indexed_vocab = utils.F2I_UNI

    vocabulary_size = len(indexed_vocab)
    languages_amount = len(utils.L2I)
    iteration_amount = 50
    learning_rate = 5e-4
    
    # Convert train examples and dev examples to the format: (lang_id, frequencies) where 
    # frequencies is a vector that represents 600 common bigrams' normalized frequencies in
    # the example's text.
    train_data = list()
    for example in train_set:
        lang, cur_bigrams = utils.L2I[example[0]], example[1]
        train_data.append((lang, get_frequencies(cur_bigrams, indexed_vocab)))
        
    dev_data = list()
    for example in dev_set:
        lang, cur_bigrams = utils.L2I[example[0]], example[1]
        dev_data.append((lang, get_frequencies(cur_bigrams, indexed_vocab)))

    params = ll.create_classifier(vocabulary_size, languages_amount)
    trained_params = train_classifier(train_data, dev_data, iteration_amount, learning_rate, params)


