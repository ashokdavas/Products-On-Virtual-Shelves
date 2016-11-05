import tensorflow as tf

#####################################################
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import MultiLabelBinarizer
import re
train_x = pd.DataFrame.from_csv('../input/train_x.csv')
test_x = pd.DataFrame.from_csv('../input/test_x.csv')
train_y = pd.DataFrame.from_csv("../input/train_y.csv")
test_y = pd.DataFrame.from_csv("../input/test_y.csv")

print ("train_test sizes are",train_x.shape,test_x.shape,train_y.shape,test_y.shape)

# hidden Layer
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        w_h = tf.Variable(tf.random_normal([n_in, n_out],mean = 0.0,stddev = 0.05))
        b_h = tf.Variable(tf.zeros([n_out]))

        self.w = w_h
        self.b = b_h
        self.params = [self.w, self.b]

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.relu(linarg)

        return self.output

# output Layer
class OutputLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        w_o = tf.Variable(tf.random_normal([n_in, n_out], mean = 0.0, stddev = 0.05))
        b_o = tf.Variable(tf.zeros([n_out]))

        self.w = w_o
        self.b = b_o
        self.params = [self.w, self.b]

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        #changed relu to sigmoid
        self.output = tf.nn.sigmoid(linarg)

        return self.output

# model
def model(x,y):
    # h_layer = HiddenLayer(input = x, n_in = 20000, n_out = 1000)
    o_layer = OutputLayer(input = x, n_in = 120, n_out = 32)

    # loss function
    out = o_layer.output()
    # modified cross entropy to binary cross entropy
    #cross_entropy = -tf.reduce_sum( (  (y_*tf.log(out + 1e-9)) + ((1-y_) * tf.log(1 - out + 1e-9)) )  , name='xentropy' )

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=out,targets=y)
    # regularization
    l2 =  tf.nn.l2_loss(o_layer.w)
    lambda_2 = 0.01

    # compute loss
    loss = cross_entropy + lambda_2 * l2

    # compute accuracy for single label classification task
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

    return loss, accuracy


train_nn = tf.placeholder(tf.float32,[None,120])
target_nn = tf.placeholder(tf.float32,[None,32])

print ("starting neural network")
loss, accuracy = model(train_nn,target_nn)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
initialize = tf.initialize_all_variables()
sess = tf.Session()
sess.run(initialize)
for ind in range(100):
    sess.run(train_step, feed_dict={train_nn: train_x, target_nn: train_y})

print(sess.run(accuracy, feed_dict={train_nn: test_x, target_nn: test_y}))