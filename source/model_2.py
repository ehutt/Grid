#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:02:13 2018

@author: elizabethhutton
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 

##############################################################################
    ######################### Data Processing ###########################

#read data into dataframes 
in_file = "TN_ABCD.txt"
target_file = "TNP1_ABCD.txt"
test_file = "ABCD_test_items.txt" 
num_trials = 1023

df_in = pd.read_csv(in_file, header = None, delim_whitespace = True)
df_target = pd.read_csv(target_file, header = None, delim_whitespace = True)
df_test_pairs = pd.read_csv(test_file, header = None, delim_whitespace = True)

#split test items into input and output 
test_in = df_test_pairs.values[:,1:3]
test_out = df_test_pairs.values[:,3:5]

#reformat data s.t. input b/w [-1,1], output b/w [0,1]
data_in = df_in.values
data_in = ((data_in - 1)/3.5)-1

target = df_target.values
target = ((target - 1)/7)

test_in = ((test_in - .1)/.35)-1
test_out = ((test_out - .1)/.7)

#get category label from input coordinates 
def get_input_label(x,y) : 
    sp = 0 
    category = np.zeros([1,4])
    if x<=sp and y <=sp:
        category = [0,1,0,0] #B
    elif x>sp and y<=sp:
        category = [0,0,0,1] #D
    elif x<=sp and y>sp:
        category = [1,0,0,0] #A
    elif x>sp and y>sp:
        category = [0,0,1,0] #C 
    else:
        print("Error, separation point")
    return category

#generate labels for input 
labels = np.zeros([num_trials, 4])
cat_names = ['A','B','C','D']
for i in range(num_trials):
    labels[i,:] = get_input_label(data_in[i,0],data_in[i,1])

#generate labels for test items (to test classification) 
n_test = test_in.shape[0]
test_labels = np.zeros([n_test, 4])
for i in range(n_test):
    test_labels[i,:] = get_input_label(test_in[i,0],test_in[i,1])



##############################################################################
    ######################### Model Design ###########################


def model(input_, target, test_input, test_target, l_rate):

    #parameters 
    n_epochs = 2500
    n_features = input_.shape[1]
    n_outputs = target.shape[1]

    #input and output placeholders 
    with tf.name_scope('input'):
        x_in = tf.placeholder(tf.float64,[None, n_features])
        y_target = tf.placeholder(tf.float64,[None, n_outputs])

    #num hidden layers and sizes 
    n_hidden_1 = 10
    n_hidden_2 = 10

    # Define weights and biases 
    W = {
        'h1': tf.Variable(tf.truncated_normal([n_features, n_hidden_1],dtype = tf.float64)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],dtype = tf.float64)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_outputs],dtype = tf.float64))
    }
    
    b = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1], dtype = tf.float64)),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2], dtype = tf.float64)),
        'out': tf.Variable(tf.truncated_normal([n_outputs],dtype = tf.float64))
    }

    #build model 
    def neural_net(x):
        
        # Hidden layers with sigmoid activation
        with tf.name_scope('layer1'):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W['h1']), b['b1']))
        with tf.name_scope('layer2'):
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W['h2']), b['b2']))
    
        # Output layer
        with tf.name_scope('output_layer'):
            out_layer = tf.add(tf.matmul(layer_2, W['out']), b['out'])
        return out_layer

    
    #generate predictions for train and test
    y_out = neural_net(input_)
    #train_pred = tf.nn.softmax(y_out)
    
    test_out = neural_net(test_input)
    #test_pred = tf.nn.softmax(test_out)
    
    errors = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(test_out, test_target)),1,keepdims=True))

    #define loss and optimizer
    with tf.name_scope('cost'):
        cost_fxn = tf.reduce_mean(tf.losses.mean_squared_error(predictions = y_out, labels = y_target))
        tf.summary.scalar('cost', cost_fxn) 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = l_rate)
    with tf.name_scope('train'):
        train_op = optimizer.minimize(cost_fxn)

    # Evaluate model accuracy 
    with tf.name_scope('train_accuracy'):    
        train_correct = tf.equal(tf.argmax(y_out, 1), tf.argmax(target, 1))
        train_acc = tf.reduce_mean(tf.cast(train_correct, tf.float32))
    
    with tf.name_scope('test_accuracy'):    
        test_correct = tf.equal(tf.argmax(test_out, 1), tf.argmax(test_target, 1))
        test_acc = tf.reduce_mean(tf.cast(test_correct, tf.float32))
        test_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions = test_out, labels = test_target))

    
    #initialize variables
    init = tf.global_variables_initializer()
    
    #save mse and accuracy info 
    cost_log = [] 
    accuracy_log = []
    test_cost_log = []
    test_accuracy_log = []
    #errors = []
    log_step = 500

    # Training
    with tf.Session() as sess:
        
        #define file writer 
        #writer = tf.summary.FileWriter('./graphs', sess.graph)
    
        #run initializer 
        sess.run(init)
        
        #train
        for i in range(n_epochs): 
            sess.run(train_op,feed_dict = {x_in: input_, y_target: target})
            
            #generate cost and accuracy info 
            cost, train_accuracy = sess.run([cost_fxn, train_acc],feed_dict = \
                                    {x_in: input_, y_target: target})
            cost_log.append(cost)
            accuracy_log.append(train_accuracy)
            
            #get accuracy on test items
            test_error,test_accuracy = sess.run([test_loss,test_acc],feed_dict = \
                                    {x_in: test_input, y_target: test_target})
            test_cost_log.append(test_error)
            test_accuracy_log.append(test_accuracy)
            
            #add to summary 
            #merged = tf.summary.merge_all()
            #summary = sess.run(merged, feed_dict = {x_in: data_in, y_target: labels})
            #writer.add_summary(summary, i)
            
            if i == 0 or (i+1) % log_step == 0:
                #print progress 
                print("Epoch " + str(i+1) + ", Loss= " + \
                      "{:.3f}".format(cost) + ", Train Accuracy= " \
                      + "{:.3f}".format(train_accuracy) + ", Test Accuracy= " \
                      + "{:.3f}".format(test_accuracy))
                
            
            #save output layer activations on last epoch
            if i == n_epochs-1: 
                output_layer = neural_net(test_in).eval()
                
                #errors.append(item_error)
                errors = errors.eval()

                
    plt.figure() 
    plt.plot(cost_log, label='Cost')
    plt.plot(test_accuracy_log, label = 'Test')
    plt.plot(accuracy_log, label = 'Train')
    plt.legend() 
    #writer.close() 
    return output_layer, errors

#category training 
output_layer,tmp = model(data_in,labels,test_in,test_labels, l_rate = 0.5)
    
#prediction after training
input_ = np.concatenate([data_in,output_layer],1)
test_input = np.concatenate([test_in,np.zeros([16,4])],1)
output, trained_errors = model(input_,target,test_input,test_out,l_rate = 0.1)

#pure prediction 
output, simple_errors = model(data_in, target,test_in,test_out,l_rate = 0.1)


    
### run models multiple times 
    
iterations = 3 
for i in range(iterations): 
    output,trained = model(input_,target,test_input,test_out,l_rate = 0.1)
    trained_errors = np.concatenate((trained_errors,trained),axis=1)
    output, simple = model(data_in, target,test_in,test_out,l_rate = 0.1)
    simple_errors = np.concatenate((simple_errors,simple),axis=1)

    
error_by_cat_trained = np.array([np.mean(trained_errors[0:4,:]),\
                                 np.mean(trained_errors[4:8,:]),\
                                 np.mean(trained_errors[8:12,:]),\
                                 np.mean(trained_errors[12:16,:])])

error_by_cat_simple = np.array([np.mean(simple_errors[0:4,:]),\
                                 np.mean(simple_errors[4:8,:]),\
                                 np.mean(simple_errors[8:12,:]),\
                                 np.mean(simple_errors[12:16,:])])
###plot results
test_cats = ('Word', 'Part Word', 'Cor. Traj.', 'Inc. Traj')
fig, ax = plt.subplots()
x_pos = np.arange(len(test_cats))
bar_width = 0.4
plt.bar(x_pos, error_by_cat_simple, -bar_width, align = 'edge',alpha=0.5, label='Simple')
plt.bar(x_pos, error_by_cat_trained, bar_width, align = 'edge',alpha=0.5, label='Trained')
plt.xticks(x_pos, test_cats)
plt.ylabel('Error')
plt.title('Test Errors by Condition')
plt.legend()
plt.show()




