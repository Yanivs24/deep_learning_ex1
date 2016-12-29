import numpy as np

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}

def softmax(x):

    # Decrease the scalar max(x) from every element in the vector to avoid
    # very large exponentials
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)

def classifier_output(x, params):

    W, b, U, b_tag = params
    
    layer1_out = np.tanh(np.dot(x, W) + b)
    hidden_out = np.dot(layer1_out, U) + b_tag
    return softmax(hidden_out)

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb,gU,gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of W
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params

    # calculate cross-entropy loss when the correct label is a 'one hot'
    y_hat = classifier_output(x, params) 
    loss = -np.log(y_hat[y]) # only the correct label will be in the sum

    #############################
    #      Calc Gradients       #
    #############################
    layer1_out = np.tanh(np.dot(x, W) + b)
    softmax_out = softmax(np.dot(layer1_out, U) + b_tag)
    der_layer1_out = 1 - layer1_out**2

    # gradient of b_tag
    gb_tag = np.copy(softmax_out)
    gb_tag[y] -= 1

    # gradient of U
    gU = np.zeros(U.shape)
    # built from layer1_out[i]*softmax_out in every row plus (-layer1_out[i]) in the y'th column
    for (i,j) in np.ndindex(gU.shape):
        gU[i,j] = layer1_out[i]*softmax_out[j] - layer1_out[i] * (j == y)

    # chain rule - first we get dloss/dz (while z = tanh(xW+b))
    dloss_dlayer1 = -U[:,y] + np.dot(U, softmax_out)
    # gradient of b
    dlayer1_db = der_layer1_out # dz/db
    gb = dloss_dlayer1 * dlayer1_db

    # gradient of W
    gW = np.zeros(W.shape) 
    for (i,j) in np.ndindex(gW.shape):
        gW[i,j] = der_layer1_out[j] * x[i] * dloss_dlayer1[j] # (dloss/dZj) * dZj/dWij

    #############################
    return loss,[gW,gb,gU,gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    # according to Xavier Glorot et al's suggestion for initialization:
    good_init = lambda n,m: np.random.uniform(-np.sqrt(6.0/(n+m)), np.sqrt(6.0/(n+m)), (n,m) \
                if (n!=1 and m!=1) else n*m)

    # MLP input (x) -> hidden
    W = good_init(in_dim, hid_dim)
    b = good_init(1, hid_dim)
    # hidden output -> MLP outputgr
    U = good_init(hid_dim, out_dim)
    b_tag = good_init(1, out_dim)
    
    params = [W,b,U,b_tag]
    return params


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b,U,b_tag = create_classifier(3,6,5)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[1]

    def _loss_and_U_grad(U):
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[2]

    def _loss_and_btag_grad(b_tag):
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[3]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0],U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])

        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_btag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)