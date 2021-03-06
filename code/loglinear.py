import numpy as np

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.

    # Decrease the scalar max(x) from every element in the vector to avoid
    # very large exponentials
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params
    
    return softmax(np.dot(x, W) + b)

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W,b = params
    # calculate cross-entropy loss when the correct label is a 'one hot'
    y_hat = classifier_output(x, params) # get softmax(xW+b)

    # debug:
    if 0 in y_hat:
        print 'probs =\n%s\n\n' % y_hat
        print 'xW+b =\n%s\n\n' % (np.dot(x, W) + b,)
        print 'x =\n%s\n\n' % x
        #print 'W =\n%s\n\n' % W
        print 'b =\n%s\n\n' % b
        exit()

    loss = -np.log(y_hat[y]) # only the correct label will be in the sum

    # Calc Gradients
    softmax_out = softmax(np.dot(x,W)+b)

    # gradient of b
    gb = np.copy(softmax_out)
    gb[y] -= 1

    # gradient of W
    gW = np.zeros(W.shape)
    # built from x[i]*softmax(xW+b) in every row plus (-x[i]) in the y'th column
    for (i,j) in np.ndindex(gW.shape):
        gW[i,j] = x[i]*softmax_out[j] - x[i] * (j == y)

    return loss,[gW,gb]

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print test2
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print test3 
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])

        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)


    
