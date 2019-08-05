#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:02:13 2018

@author: elizabethhutton
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
import pickle as pkl 

##############################################################################
    ######################### Data Processing ###########################

#read data into dataframes 
in_file = "data/TN_ABCD.txt"
target_file = "data/TNP1_ABCD.txt"
test_file = "data/ABCD_test_items.txt" 
num_trials = 1023

df_in = pd.read_csv(in_file, header = None, delim_whitespace = True)
df_out = pd.read_csv(target_file, header = None, delim_whitespace = True)
df = pd.concat([df_in,df_out],axis=1)

# Part 1: Training the Model
#divide data into train (80%) and test (20%) sets
train_df = df.sample(frac=0.8,random_state=200)
test_df = df.drop(train_df.index)

#reformat data s.t. input/outputs b/w [-1,1] 
#split into input and output sets
train = ((train_df.values - 1)/3.5)-1
train_in = train[:,0:2]
train_out = train[:,2:4]

test = ((test_df.values - 1)/3.5)-1
test_in = test[:,0:2]
test_out = test[:,2:4]

# Part 2: Simulating SL Experiment
#create training and test sets for simulation 
#train on full data set
all_in = ((df_in.values - 1)/3.5)-1
all_out = ((df_out.values - 1)/3.5)-1
#test pairs for comparison to human subs 
df_test_pairs = pd.read_csv(test_file, header = None, delim_whitespace = True)
#split data into input and target outputs 
#scale inputs/outputs to be within [-1,1]
items_in = ((df_test_pairs.values[:,1:3] - .1)/.35)-1
items_out = ((df_test_pairs.values[:,3:5] - .1)/.35)-1


##############################################################################
    ######################### Model Design ###########################

def model(train_in, train_out, test_in, test_out):
    #parameters 
    n_epochs = 2500
    l_rate = 0.1
    n_features = train_in.shape[1]
    n_outputs = train_out.shape[1]
    
    #input and output placeholders 
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float64,[None, n_features])
        Y_target = tf.placeholder(tf.float64,[None, n_outputs])
    
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
        
        # Hidden layers with tanh activation
        with tf.name_scope('layer1'):
            layer_1 = tf.tanh(tf.add(tf.matmul(x, W['h1']), b['b1']))
        with tf.name_scope('layer2'):
            layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, W['h2']), b['b2']))
    
        # Output layer
        with tf.name_scope('output_layer'):
            out_layer = tf.add(tf.matmul(layer_2, W['out']), b['out'])
        return out_layer
    
    #get predictions from output layer of network
    y_hat = neural_net(X)
    
    #define loss and training optimizer
    cost = tf.reduce_mean(tf.square(y_hat-Y_target))
    train = tf.train.GradientDescentOptimizer(l_rate).minimize(cost)  
    
    #save mse and accuracy info 
    train_cost_log = [] 
    test_cost_log = []
    log_step = 500
    
    #initialize variables
    init = tf.global_variables_initializer()
    
    # Training
    with tf.Session() as sess:
    
        #run initializer 
        sess.run(init)
        
        #train
        for i in range(n_epochs): 
            sess.run([cost,train],feed_dict = {X: train_in, Y_target: train_out})
            
            #generate and save cost for train and test 
            c_train = sess.run(cost,feed_dict = {X: train_in, Y_target: train_out})
            train_cost_log.append(c_train)
            c_test = sess.run(cost,feed_dict = {X: test_in, Y_target: test_out})
            test_cost_log.append(c_test)
            
            #report train/test results every 500 epochs
            if i == 0 or (i+1) % log_step == 0:
                #print progress 
                print("Epoch " + str(i+1) + ", Train MSE= " \
                      + "{:.3f}".format(c_train) + ", Test MSE= " \
                      + "{:.3f}".format(c_test))
            #save and return final test_mse 
            if i == n_epochs-1: 
                test_preds = neural_net(test_in).eval()
                
    
    #plot train vs test mse 
    plt.figure() 
    plt.plot(test_cost_log, label = 'Test')
    plt.plot(train_cost_log, label = 'Train')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.legend() 
    
    return test_preds


#train model using entire dataset (split into train and test to evaluate)
training_preds = model(train_in,train_out,test_in,test_out)

##############################################################################
    ######################### Simulation ###########################
    
    
#extract test_mse from last epoch training
testing_preds = model(all_in,all_out,items_in,items_out)

#evaluate accuracy by test category 
test_mse = (np.square(testing_preds - items_out)).mean(axis=1)
error_by_cat = np.array([test_mse[0:4],\
                                 test_mse[4:8],\
                                 test_mse[8:12],\
                                 test_mse[12:16]])
filename = 'simple_predictions.pickle'
with open(filename, 'wb') as f:
    pkl.dump(testing_preds, f)
filename = 'error_by_cat_simple.pickle'
with open(filename, 'wb') as f:
    pkl.dump(error_by_cat, f)
    
###plot results
test_cats = ('Word', 'Part Word', 'Cor. Traj.', 'Inc. Traj')
fig, ax = plt.subplots()
bp = ax.boxplot(error_by_cat.transpose())
ax.set_xticklabels(test_cats)
plt.ylabel('MSE')
plt.title('Model Error by Test Condition')
plt.show()
fig.savefig('fig1.png', bbox_inches='tight')

    
