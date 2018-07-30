from load import mnist
import numpy as np
import pylab 
import time

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

"""
DATASET USED: MNIST database

This file trains a three-layer convolutional neural network (CNN) model using different learning methods, stochastic gradient descent (sgd), 
stochastic gradient descrent with momentum term (sgd with momentum) and RMSProp. 

By defining the variable 'method' flobally, the file can train and test model using different learning methods:

method = 'sgd', 'sgd with momentum' or 'RMSProp'
"""
 
np.random.seed(10)

#defining paramters
batch_size = 128
noIters = 100 #noIters = 100 is required or sufficient
learning_rate = 0.05
learning_rate_RMSProp = 0.001 #Learning rate for RMSProp algorithm
beta = 1e-4 #decay parameter
gamma = 0.1 #mommentum parameter
rho = 0.9 #RMSProp 
e = 1e-6 #RMSProp


def init_weights_bias4(filter_shape, d_type): # e.g. filer_shape = (15,1,9,9)
    fan_in = np.prod(filter_shape[1:]) # e.g. 1*9*9
    fan_out = filter_shape[0] * np.prod(filter_shape[2:]) # e.g. 15 * 9*9
     
    bound = np.sqrt(6. / (fan_in + fan_out)) # draw from uniform distr.
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type) # e.g. transpose(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_weights_bias2(filter_shape, d_type): # e.g. filer_shape = (15*10*10,10)
    fan_in = filter_shape[1] # e.g. 10
    fan_out = filter_shape[0] # e.g. 15*10*10
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)


def model(X, w1, b1, w2, b2, w3, b3, w4, b4): 
    y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
    pool_dim1 = (2, 2) #pooling window of size 2x2
    o1 = pool.pool_2d(y1, pool_dim1) # e.g. 15*10*10

    y2 = T.nnet.relu(conv2d(o1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
    pool_dim2 = (2, 2)
    o2 = pool.pool_2d(y2, pool_dim2) # e.g 20*3*3
    
    o3 = T.flatten(o2, outdim=2)
    y3 = T.nnet.relu(T.dot(o3, w3) + b3) #fully connected layer
    
    pyx = T.nnet.softmax(T.dot(y3, w4) + b4)
    
    return y1, o1, y2, o2, pyx

def sgd(cost, params, lr=learning_rate, decay = beta):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - (g + decay*p) * lr])
    return updates

def sgd_momentum(cost, params, lr=learning_rate, decay=beta, momentum=gamma):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value()*0.)
        v_new = momentum*v - (g + decay*p) * lr 
        updates.append([p, p + v_new])
        updates.append([v, v_new])
    return updates

def RMSprop(cost, params, lr=learning_rate_RMSProp, decay=beta, rho = rho, epsilon = e):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * (g+ decay*p)))
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels
    
trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)


trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]


X = T.tensor4('X')
Y = T.matrix('Y')

#define the number of feature maps for each convolution layer
num_filters1 = 15 #feature maps for C1
num_filters2 = 20 #feature maps for C2
fconnect_size = 100 #fully connected layer of size 100
smax_size = 10 # size of softmax layer

w1, b1 = init_weights_bias4((num_filters1, 1, 9, 9), X.dtype) #weights and bias for C1
w2, b2 = init_weights_bias4((num_filters2, 15, 5, 5), X.dtype) # weights and bias for C2
w3, b3 = init_weights_bias2((num_filters2*3*3, fconnect_size), X.dtype)
w4, b4 = init_weights_bias2((fconnect_size, smax_size), X.dtype) #weights and bias for softmax layer

y1, o1, y2, o2, py_x  = model(X, w1, b1, w2, b2, w3, b3, w4, b4)

y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w1, b1, w2, b2, w3, b3, w4, b4]


#differentiate updates with different algorithms used
updates1 = sgd(cost, params)
updates2 = sgd_momentum(cost, params)
updates3 = RMSprop(cost, params)

train1 = theano.function(inputs=[X, Y], outputs=cost, updates=updates1, allow_input_downcast=True)
train2 = theano.function(inputs=[X, Y], outputs=cost, updates=updates2, allow_input_downcast=True)
train3 = theano.function(inputs=[X, Y], outputs=cost, updates=updates3, allow_input_downcast=True)


predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test = theano.function(inputs = [X], outputs=[y1, o1], allow_input_downcast=True)


if method == 'sgd':
    print(method)
    a1 = [] #record test accuracy for each iteration
    c1 = [] #record train cost for each iteration

    for i in range(noIters):
        trX, trY = shuffle_data (trX, trY)
        teX, teY = shuffle_data (teX, teY)
        c=[]
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train1(trX[start:end], trY[start:end]))
        a1.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
        c1.append(np.mean(c, dtype='float64'))
        print("%d test accuracy: %.4f, train cost: %.4f" %(i+1, a1[i], c1[i]))

if method == 'sgd with momentum':
    print(method)
    a2 = [] #record test accuracy for each iteration
    c2 = [] #record train cost for each iteration

    for i in range(noIters):
        trX, trY = shuffle_data (trX, trY)
        teX, teY = shuffle_data (teX, teY)
        c=[]
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train2(trX[start:end], trY[start:end]))
        a2.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
        c2.append(np.mean(c, dtype='float64'))
        print("%d test accuracy: %.4f, train cost: %.4f" %(i+1, a2[i], c2[i]))

if method == 'RMSprop':
    print (method)
    a3 = [] #record test accuracy for each iteration
    c3 = [] #record train cost for each iteration

    for i in range(noIters):
        trX, trY = shuffle_data (trX, trY)
        teX, teY = shuffle_data (teX, teY)
        c=[]
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train3(trX[start:end], trY[start:end]))
        a3.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
        c3.append(np.mean(c, dtype='float64'))
        print("%d test accuracy: %.4f, train cost: %.4f" %(i+1, a3[i], c3[i]))
"""
w = w1.get_value()
pylab.figure()
pylab.gray()
for i in range(w.shape[0]):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(w[i,:,:,:].reshape(9,9))
#pylab.title('filters learned')
#pylab.savefig('figure_2a_2.png')
"""

ind = np.random.randint(low=0, high=2000)
convolved, pooled = test(teX[ind:ind+1,:])

print('input image...')
pylab.figure()
pylab.gray()
pylab.axis('off'); pylab.imshow(teX[ind,:].reshape(28,28))
#pylab.title('input image')
pylab.savefig(method + '_input_image1.png')

print('convolved feature map...')
pylab.figure()
pylab.gray()
for i in range(convolved.shape[1]):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved[0,i,:])
#pylab.title('convolved feature maps')
pylab.savefig(method + '_convolved_image1.png')

print('pooled feature map...')
pylab.figure()
pylab.gray()
for i in range(pooled.shape[1]):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled[0,i,:])
#pylab.title('pooled feature maps')
pylab.savefig(method + '_pooled_image1.png')

pylab.show()

np.random.seed(12)

ind = np.random.randint(low=0, high=2000)
convolved, pooled = test(teX[ind:ind+1,:])

print('input image...')
pylab.figure()
pylab.gray()
pylab.axis('off'); pylab.imshow(teX[ind,:].reshape(28,28))
#pylab.title('input image')
pylab.savefig(method + '_input_image2.png')

print('convolved feature map...')
pylab.figure()
pylab.gray()
for i in range(convolved.shape[1]):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved[0,i,:])
#pylab.title('convolved feature maps')
pylab.savefig(method + '_convolved_image2.png')

print('pooled feature map...')
pylab.figure()
pylab.gray()
for i in range(pooled.shape[1]):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled[0,i,:])
#pylab.title('pooled feature maps')
pylab.savefig(method + '_pooled_image3.png')

pylab.show()