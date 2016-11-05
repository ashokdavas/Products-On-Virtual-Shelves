from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re
#Read either from file after feature selection processing or before that
test = pd.read_csv("../input/test_data_processed.csv")
train =pd.read_csv("../input/train_data_processed.csv")
target =pd.read_csv("../input/target_data_processed.csv")
index_test = test.index

####################################################
# TARGET CLASSES FORMATION INTO ONE HOT PARTITION

train_actual = pd.DataFrame.from_csv('../input/train.tsv', sep='\t', header=0)
target = train["tag"]

##find iteger classes from target column
classes = set()
tags = []
for x in target:
    y = map(int, re.findall(r'\d+', x))
    tags.append(y)
    classes.update(y)
int_target = np.array(tags)

#Binarize the target classes into one hot partition format
#MultiLabelBinarizer also inverse transform the result.

mlb = MultiLabelBinarizer()
target = mlb.fit_transform(int_target)
target = pd.DataFrame(target)

####################################################
current_pos = 0
total_size = 10593
def next_batch(size):
    global current_pos
    new_end_pos = current_pos + size
    if(new_end_pos>=total_size):
        new_end = new_end_pos % (total_size)
        tx = np.concatenate((train[current_pos:total_size,:],train[:new_end,:]),axis=0)
        ty = np.concatenate((target[current_pos:total_size,:],target[:new_end,:]),axis=0)
        current_pos = new_end
        return tx,ty
    else:
        tx, ty = np.array(train[current_pos:new_end_pos,:]),np.array(target[current_pos:new_end_pos,:])
        current_pos = new_end_pos
        # print ("tx is ",tx[:2,:])
        return tx,ty

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 1500
batch_size = 100
display_step = 100

# Network Parameters
n_hidden_1 = 120 # 1st layer number of features
n_hidden_2 = 120 # 2nd layer number of features
n_input = 120 # MNIST data input (img shape: 28*28)
n_classes = 32 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

output_s = tf.nn.sigmoid(pred)
output_b = tf.round(output_s)
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("current cost",c)
            # print ("current pred",p[0:5,:])
            # print ("current batch",batch_x[0:5,:])
            # print("Epoch:", '%04d' % (epoch+1), "cost=", \
            #     "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: train, y: target}))

    actual_output = sess.run(pred,feed_dict={x:test})
    print ("output actual ",actual_output)
    sigmoid_output = sess.run(output_s,feed_dict={pred:actual_output})
    binary_output = sess.run(output_b,feed_dict={output_s:sigmoid_output})
    print ("output sigmoid",sigmoid_output)
    print ("output binary",binary_output)

    predicted = mlb.inverse_transform(binary_output)
    # print (predicted)
    print("len predicted", len(predicted))

    result = []
    for idx, x in enumerate(predicted):
        x = list(x)
        if (len(x) == 0):
            x.append(4483)
        # y = map(int, re.findall(r'\d+', x))
        # print y
        result.append(x)
    result = np.array(result)
    print
    result.shape, result
    to_write = {"item_id": index_test, "tag": result}
    # pd.DataFrame(to_write).to_csv()
    pd.DataFrame(to_write).to_csv("../input/mlp_own.tsv", index=False, sep="\t")
