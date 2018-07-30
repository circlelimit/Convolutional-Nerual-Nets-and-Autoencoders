from load import mnist
import numpy as np

import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

"""
DATASET USED: MNIST database

This file trains a 3-layer stacked denoising autoencoder using layer-wise training method 
as well as training a 5-layer feed forward neural network which was initialised by the 3 hidden 
layers of the stacked autoencoder and an additional softmax layer using back-propagation. 

Two learning methods were used: Stochastic gradient descent (sgd) and stochastic gradient descent with momentum term (sgd_momentum)
Two cost functions were considered: cross entropy and cost entropy with sparsity constraint. 

By defining the variable 'method' and 'sparsity' flobally, the file can train and test model using different learning methods
or whether sparsity constraint is included:

method = 'sgd' or 'sgd_momentum'
Sparsity_ind = False or True
"""
corruption_level=0.1
training_epochs = 25
learning_rate = 0.1
batch_size = 128

gamma = 0.1 #momentum parameter
beta = 0.5 #penality term
rho = 0.05 # sparsity term

# 1 encoder, decoder and a softmax layer

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)

def sgd(cost, params, lr=learning_rate, decay = 0.0001):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - (g + decay*p) * lr])
    return updates

def sgd_momentum(cost, params, lr=learning_rate, decay=0.0001, momentum=gamma):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value()*0.)
        v_new = momentum*v - (g + decay*p) * lr 
        updates.append([p, p + v_new])
        updates.append([v, v_new])
    return updates

def define_cost_da(pred_y, target_y, h, sparsity_ind=sparsity_ind, penalty = beta, sparsity = rho):
    """
    pred_y : predicted output in autoencoder
    target_y : true value of output in autoencoder
    h : the hidden layer activation
    sparsity_ind : true if sparsity term is included; vice versa.
    penality : penality term
    sparsity : sparsity term
    """
    
    if sparsity_ind == False: 
        cost = - T.mean(T.sum(target_y * T.log(pred_y) + (1 - target_y) * T.log(1 - pred_y), axis=1))
    else: 
        cost = - T.mean(T.sum(target_y * T.log(pred_y) + (1 - target_y) * T.log(1 - pred_y), axis=1)) \
        + penalty*T.shape(h)[1]*(sparsity*T.log(sparsity) + (1-sparsity)*T.log(1-sparsity)) \
        - penalty*sparsity*T.sum(T.log(T.mean(h, axis=0)+1e-6)) \
        - penalty*(1-sparsity)*T.sum(T.log(1-T.mean(h, axis=0)+1e-6))
    
    return cost
        
trX, teX, trY, teY = mnist() # use all data
#trX, trY = trX[:1000], trY[:1000]
#teX, teY = teX[:200], teY[:200]

x = T.fmatrix('x')  
d = T.fmatrix('d')


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

# Initialise weights and bias for auto-encoder
# Encoding layers
W1 = init_weights(28*28, 900)
b1 = init_bias(900)
W2 = init_weights(900, 625)
b2 = init_bias(625)
W3 = init_weights(625, 400)
b3 = init_bias(400)
# Decoding layers
W3_prime = W3.transpose()
b3_prime = init_bias(625)
W2_prime = W2.transpose()
b2_prime = init_bias(900)
W1_prime = W1.transpose()
b1_prime = init_bias(28*28)

# Initialise weights and bias for softmax layer (Image Classification)
#W2 = init_weights(900, 10)
#b2 = init_bias(10)

# Corrupted x
tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x
# Stack 1
h1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z_x = T.nnet.sigmoid(T.dot(h1, W1_prime) + b1_prime)
#cost1 = - T.mean(T.sum(x * T.log(z_x) + (1 - x) * T.log(1 - z_x), axis=1))
cost1 = define_cost_da(z_x, x, h1)
params1 = [W1, b1, b1_prime]
#grads1 = T.grad(cost1, params1)

# Update using Gradient Descent
# Create list of updates
"""
updates1 = [(param1, param1 - learning_rate * grad1)
           for param1, grad1 in zip(params1, grads1)]
"""
if method == 'sgd_momentum': 
    updates1 = sgd_momentum(cost1, params1)
else: 
    updates1 = sgd(cost1, params1)
    
train_da1 = theano.function(inputs=[x], outputs = cost1, updates = updates1, allow_input_downcast = True)
test_da1 = theano.function(inputs = [x], outputs =[h1, z_x], allow_input_downcast = True)

# Stack 2
h2 = T.nnet.sigmoid(T.dot(h1, W2) + b2)
z1 = T.nnet.sigmoid(T.dot(h2, W2_prime) + b2_prime)
#cost2 = - T.mean(T.sum(h1 * T.log(z1) + (1 - h1) * T.log(1 - z1), axis=1))
cost2 = define_cost_da(z1, h1, h2)
params2 = [W2, b2, b2_prime]
#grads2 = T.grad(cost2, params2)
"""
updates2 = [(param2, param2 - learning_rate * grad2)
           for param2, grad2 in zip(params2, grads2)]
"""

if method == 'sgd_momentum': 
    updates2 = sgd_momentum(cost2, params2)
else: 
    updates2 = sgd(cost2, params2)
    
train_da2 = theano.function(inputs=[h1], outputs = cost2, updates = updates2, allow_input_downcast = True)
test_da2 = theano.function(inputs = [h1], outputs =[h2, z1], allow_input_downcast = True)

# Stack 3
h3 = T.nnet.sigmoid(T.dot(h2, W3) + b3)
z2 = T.nnet.sigmoid(T.dot(h3, W3_prime) + b3_prime)
#cost3 = - T.mean(T.sum(h2 * T.log(z2) + (1 - h2) * T.log(1 - z2), axis=1))
cost3 = define_cost_da(z2, h2, h3)
params3 = [W3, b3, b3_prime]
#grads3 = T.grad(cost3, params3)
"""
updates3 = [(param3, param3 - learning_rate * grad3)
           for param3, grad3 in zip(params3, grads3)]
"""
if method == 'sgd_momentum': 
    updates3 = sgd_momentum(cost3, params3)
else: 
    updates3 = sgd(cost3, params3)
    
train_da3 = theano.function(inputs=[h2], outputs = cost3, updates = updates3, allow_input_downcast = True)
test_da3 = theano.function(inputs = [h2], outputs =[h3, z2], allow_input_downcast = True)


#soft-max layer for feedforward neural network
W4= init_weights(400, 10)
b4 = init_bias(10)

p_y2 = T.nnet.softmax(T.dot(h3, W4)+b4)
y2 = T.argmax(p_y2, axis=1)
cost4 = T.mean(T.nnet.categorical_crossentropy(p_y2, d))

params4 = [W1, b1, W2, b2, W3, b3, W4, b4]
#grads2 = T.grad(cost2, params2)
"""
updates4 = [(param4, param4 - learning_rate * grad4)
           for param4, grad4 in zip(params4, grads4)]
"""

if method == 'sgd_momentum': 
    updates4 = sgd_momentum(cost4, params4)
else: 
    updates4 = sgd(cost4, params4)
    
train_ffn = theano.function(inputs=[x, d], outputs = cost4, updates = updates4, allow_input_downcast = True)
test_ffn = theano.function(inputs=[x], outputs = y2, allow_input_downcast=True)


# Train Denoising Auto-encoder 1
print('training dae1 ...')
d_da1 = []
for epoch in range(training_epochs):
    # go through trainng set
    c_da1 = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c_da1.append(train_da1(trX[start:end]))
    d_da1.append(np.mean(c_da1, dtype='float64'))
    print(d_da1[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d_da1)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.title('Cost Entropy of Denoising Auto-encoder 1')
pylab.savefig(method + '_figure_cross_entropyda1.png')
pylab.show()

H1_new, Z_X = test_da1(trX)

# Train Denoising Auto-encoder 2
print('training dae2 ...')
d_da2= []
for epoch in range(training_epochs):
    # go through trainng set
    c_da2 = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c_da2.append(train_da2(H1_new[start:end]))
    d_da2.append(np.mean(c_da2, dtype='float64'))
    print(d_da2[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d_da2)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.title('Cost Entropy of Denoising Auto-encoder 2')
pylab.savefig(method +'_figure_cross_entropyda2.png')

pylab.show()

H2_new, Z_1 = test_da2(H1_new)

# Train Denoising Auto-encoder 3
print('training dae3 ...')
d_da3 = []
for epoch in range(training_epochs):
    # go through trainng set
    c_da3 = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c_da3.append(train_da3(H2_new[start:end]))
    d_da3.append(np.mean(c_da3, dtype='float64'))
    print(d_da3[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d_da3)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.title('Cost Entropy of Denoising Auto-encoder 3')
pylab.savefig(method+'_figure_cross_entropyda3.png')
pylab.show()

H3_new, Z_2 = test_da3(H2_new)

pylab.figure()
# Plots of first 100 weights in encoding layer 1
w1 = W1.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))
pylab.savefig(method+ '_figure_w1.png')

# Plots of first 100 weights in encoding layer 2
w2 = W2.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w2[:,i].reshape(30,30))
pylab.savefig(method+'_figure_w2.png')

# Plots of first 100 weights in encoding layer 3
w3 = W3.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w3[:,i].reshape(25,25))
pylab.savefig(method+'_figure_w3.png')

pylab.show()


#plot reconstructed images and hidden layer activation using 100 representative test images
te_h1, te_z_x= test_da1(teX[:100])
te_h2, te_z_1 = test_da2(te_h1)
te_h3, te_z_2 = test_da3(te_h2)

y =T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(h3, W3_prime)+ b3_prime), W2_prime)+b2_prime), W1_prime)+b1_prime)

predict = theano.function(inputs = [h3], outputs = y, allow_input_downcast = True)

te_y = predict(te_h3)

print('reconstructed images...')
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(te_y[i,:].reshape(28,28))
pylab.savefig(method+'_figure_reconstructed_image.png')
pylab.show()


print('original images...')
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(teX[:100][i,:].reshape(28,28))
pylab.savefig(method+'_figure_original_image.png')
pylab.show()


print('1st hidden layer activation...')
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(te_h1[i,:].reshape(30,30))
pylab.savefig(method+'_figure_1st_layer_hidden.png')
pylab.show()

print('2nd hidden layer activation...')
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(te_h2[i,:].reshape(25,25))
pylab.show()
pylab.savefig(method+'_figure_2nd_layer_hidden.png')

print('3rd hidden layer activation...')   
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(te_h3[i,:].reshape(20,20))   
pylab.savefig(method+'_figure_3rd_layer_hidden.png')
pylab.show()

#train the softmax layer and the entire FFN using weights and biases initialized by part i
print('\ntraining ffn ...')
d, a = [], []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_ffn(trX[start:end], trY[start:end]))
    d.append(np.mean(c, dtype='float64'))
    a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
    print(a[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.title('Cross-entropy against epochs')
pylab.savefig(method+'_figure_100_test_cross_entropy.png')

pylab.figure()
pylab.plot(range(training_epochs), a)
pylab.xlabel('iterations')
pylab.ylabel('test accuracy')
pylab.title('Test accuracy against epochs')
pylab.savefig(method+'_figure_100_test_test_accuracy.png')

pylab.show()