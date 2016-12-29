import numpy as np

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}

def softmax(x):

    # Decrease the scalar max(x) from every element in the vector to avoid
    # very large exponentials
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)

def classifier_output(x, params):
    
    params_pairs = [tuple(params[i:i+2]) for i in range(0,len(params),2)]
    
    # run throgh all layers except last one
    layer_out = x
    for W,b in params_pairs[:-1]:
        layer_out = np.tanh(np.dot(layer_out, W) + b)

    # return the softmax of the linear manipulation of the last layer
    last_W, last_b = params_pairs[-1]
    return softmax(np.dot(layer_out, last_W) + last_b)


def get_layer_values(x, params, layer):
    
    params_pairs = [tuple(params[i:i+2]) for i in range(0,len(params),2)]
    
    # run throgh all layers except last one
    layer_out = np.array(x)
    for W,b in params_pairs[:layer]:
        layer_out = np.tanh(np.dot(layer_out, W) + b)

    return layer_out

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    # calculate cross-entropy loss when the correct label is a 'one hot'

    y_hat = classifier_output(x, params) 
    loss = -np.log(y_hat[y]) # only the correct label will be in the sum

    #############################
    #      Calc Gradients-Old  #
    #############################
    # layer1_out = np.tanh(np.dot(x, W) + b)
    # softmax_out = softmax(np.dot(layer1_out, U) + b_tag)
    # der_layer1_out = 1 - layer1_out**2

    # # gradient of b_tag
    # gb_tag = np.copy(softmax_out)
    # gb_tag[y] -= 1

    # # gradient of U
    # gU = np.zeros(U.shape)
    # # built from layer1_out[i]*softmax_out in every row plus (-layer1_out[i]) in the y'th column
    # for (i,j) in np.ndindex(gU.shape):
    #     gU[i,j] = layer1_out[i]*softmax_out[j] - layer1_out[i] * (j == y)

    # # chain rule - first we get dloss/dz (while z = tanh(xW+b))
    # dloss_dlayer1 = -U[:,y] + np.dot(U, softmax_out)
    # # gradient of b
    # dlayer1_db = der_layer1_out # dz/db
    # gb = dloss_dlayer1 * dlayer1_db

    # # gradient of W
    # gW = np.zeros(W.shape) 
    # for (i,j) in np.ndindex(gW.shape):
    #     gW[i,j] = der_layer1_out[j] * x[i] * dloss_dlayer1[j] # (dloss/dZj) * dZj/dWij


    #############################
    #      Calc MLPN Gradients  #
    #############################
    grads = list()

    params_pairs = [tuple(params[i:i+2]) for i in range(0,len(params),2)]
    layers_amount = len(params_pairs)

    softmax_out = classifier_output(x, params)
    z1_out = get_layer_values(x, params, layers_amount-1)

    # We define the params for the last layer as W1,b1 and the previous
    # layer output as z1 (and so on.. e.g. Z1(W2,b2,Z2)= tanh(Z2W2+b2))
    W1, b1 = params_pairs[-1]
    inner_params = params_pairs[:-1]
    inner_params.reverse()
    # The last layer (softmax layer) is different so we will calc 
    # its gradients here - assume loss(W1,b1,z1) and we calc dloss / dz1
    dL_dz1 = -W1[:,y] + np.dot(W1, softmax_out)
    dL_db1 = np.copy(softmax_out)
    dL_db1[y] -= 1
    dL_dW1 = np.zeros(W1.shape)
    for (i,j) in np.ndindex(dL_dW1.shape):
        dL_dW1[i,j] = z1_out[i]*softmax_out[j] - z1_out[i] * (j == y)

    # add reads (in the end will be reversed)
    grads.append(dL_db1)
    grads.append(dL_dW1)
    
    # init for loop
    current_dz = dL_dz1
    current_layer = layers_amount - 2
    for W,b in inner_params:
        # get input of layer
        layer_in = get_layer_values(x, params, current_layer)
        grad_tanh = 1 - (np.tanh(np.dot(layer_in, W) + b))**2

        # ## debug ##
        # print type(layer_in)
        # print type(grad_tanh)
        # print 'current layer: %s' % str(current_layer)
        # print 'current W size: %s' % str(W.shape)
        # print 'current dz size: %s' % str(current_dz.shape)
        # ## end debug ##

        # calc currend layer gradients (befor chain rule)
        dz_db = grad_tanh
        dz_dw = np.dot(layer_in.reshape(len(layer_in),1), grad_tanh.reshape(1,len(grad_tanh))) # should build matrix
        grads.append(current_dz*dz_db) # multiply element-element
        grads.append(current_dz*dz_dw) # for each column - multiply column j by current_dz[j]

        # chain rule - multipy by next gradient (dZk/DdZk-1)
        dZ_dprevZ = grad_tanh * W  # a matrix - multiply each column by grad_tanh[i]
        # now build the gradient dL/dZk - vector of corrent gradient from the end of the net
        # until our layer
        current_dz = np.dot(dZ_dprevZ, current_dz)  
        current_layer -= 1

    # grads are in reverse order now
    grads.reverse()

    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """

    # according to Xavier Glorot et al's suggestion for initialization:
    good_init = lambda n,m: np.random.uniform(-np.sqrt(6.0/(n+m)), np.sqrt(6.0/(n+m)), (n,m) \
                if (n!=1 and m!=1) else n*m)

    # create params - random matrix and vector for every layer
    params = list()
    for i in range(len(dims)-1):
        in_dim = dims[i]
        out_dim = dims[i+1]
        params.append(good_init(in_dim, out_dim))
        params.append(good_init(1, out_dim))

    return params

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    dims = [20,30,40,10]
    params = create_classifier(dims)

    def _loss_and_p_grad(p):
        ''' General function - return loss and the gradients with respect to
            parameter p'''
        params_to_send = np.copy(params)
        par_num = 0
        for i in range(len(params)):
            if p.shape == params[i].shape:
                params_to_send[i] = p
                par_num = i

        loss,grads = loss_and_gradients(range(dims[0]),0, params_to_send)
        return loss,grads[par_num]

    for _ in xrange(10):
        my_params = create_classifier(dims)
        for pa in my_params:
                gradient_check(_loss_and_p_grad, pa)
        