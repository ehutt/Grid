#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:22:25 2018

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
tf.set_random_seed(1)

df_in = pd.read_csv(in_file, header = None, delim_whitespace = True)
df_out = pd.read_csv(target_file, header = None, delim_whitespace = True)
df = pd.concat([df_in,df_out],axis=1)

#get category label from input coordinates 
def get_input_label(x,y) : 
    sp = 4.5 
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
    labels[i,:] = get_input_label(df_in.values[i,0],df_in.values[i,1])
labels_df = pd.DataFrame(labels)

# Part 1: Training the Model
#divide data into train (80%) and test (20%) sets
train_df = df.sample(frac=0.8,random_state=200)
train_labels = labels_df.iloc[train_df.index]
test_df = df.drop(train_df.index)
test_labels = labels_df.drop(train_df.index)

#reformat input/output range
train = ((train_df.values - 1)/3.5)-1
#train = (train_df.values - 1)/7
train_in = train[:,0:2]
train_output = train[:,2:4]
test = ((test_df.values - 1)/3.5)-1
#test = (test_df.values - 1)/7
test_in = test[:,0:2]
test_output = test[:,2:4]  
    
# Part 2: Simulating SL Experiment
#create training and test sets for simulation 
#train on full data set
all_in = ((df_in.values - 1)/3.5)-1
all_out = ((df_out.values - 1)/3.5)-1
#all_in = (df_in.values - 1)/7
#all_out = (df_out.values - 1)/7
#test pairs for comparison to human subs 
df_test_pairs = pd.read_csv(test_file, header = None, delim_whitespace = True)
#scale inputs/outputs to be within [0,1]
items_in = df_test_pairs.values[:,1:3]*10
items_out = df_test_pairs.values[:,3:5]*10
n_test = 16
item_labels = np.zeros([n_test, 4])
for i in range(n_test):
    item_labels[i,:] = get_input_label(items_in[i,0],items_in[i,1])
items_in = ((items_in - 1)/3.5)-1
items_out = ((items_out - 1)/3.5)-1
#items_in = (items_in - .1)/.7
#items_out = (items_out - .1)/.7


##############################################################################
  ######################### Training Model Design ###########################


def train(train_in, train_out, test_in, test_out,PLOT):
    
    #parameters 
    n_epochs = 2500
    l_rate = 0.1
    n_features = train_in.shape[1]
    n_outputs = train_out.shape[1]
    tf.set_random_seed(1)
    
    #input and output placeholders 
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float64,[None, n_features])
        Y_target = tf.placeholder(tf.float64,[None, n_outputs])
    
    #num hidden layers and sizes 
    n_hidden_1 = 45
    n_hidden_2 = 20

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
        
        # Hidden layers with  activation
        with tf.name_scope('layer1'):
            layer_1 = tf.tanh(tf.add(tf.matmul(x, W['h1']), b['b1']))
            #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W['h1']), b['b1']))
        with tf.name_scope('layer2'):
            layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, W['h2']), b['b2']))
            #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W['h2']), b['b2']))
    
        # Output layer
        with tf.name_scope('output_layer'):
            out_layer = tf.nn.softmax(tf.add(tf.matmul(layer_2, W['out']), b['out']))
        return out_layer
    
    #get predictions from output layer of network
    y_hat = neural_net(X)

    
    cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions = y_hat, labels = Y_target))
    train = tf.train.GradientDescentOptimizer(l_rate).minimize(cost) 
    correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    

    #save mse and accuracy info 
    train_cost_log = [] 
    test_cost_log = []
    train_acc_log = [] 
    test_acc_log = []
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
            
            #generate and save cost/accuracy for train/test 
            c_train,a_train = sess.run([cost,accuracy],feed_dict = {X: train_in, Y_target: train_out})
            train_cost_log.append(c_train)
            train_acc_log.append(a_train)
            c_test,a_test = sess.run([cost,accuracy],feed_dict = {X: test_in, Y_target: test_out})
            test_cost_log.append(c_test)
            test_acc_log.append(a_test)
            
            #report train/test results every 500 epochs
            if i == 0 or (i+1) % log_step == 0:
                #print progress 
                print("Epoch " + str(i+1) + ", Train MSE= " \
                      + "{:.3f}".format(c_train) + ", Test MSE= " \
                      + "{:.3f}".format(c_test))
            #save and return final test_mse 
            if i == n_epochs-1: 
                #extract output activations over entire data set
                output_layer = neural_net(train_in).eval()
            
    if PLOT == True:            
        plt.figure() 
        #plt.plot(cost_log, label='Cost')
        plt.plot(test_acc_log, label = 'Test')
        plt.plot(train_acc_log, label = 'Train')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend() 
        
        plt.figure() 
        #plt.plot(cost_log, label='Cost')
        plt.plot(test_cost_log, label = 'Test')
        plt.plot(train_cost_log, label = 'Train')
        plt.ylabel('MSE')
        plt.xlabel('Epochs')
        plt.legend() 
    
    return output_layer

##############################################################################
    ######################### SL Model Design ###########################
    
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
    n_hidden_1 = 20
    n_hidden_2 = 15
    
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


##############################################################################
    ######################### Evaluate Models ###########################
    
#establish network is learning on test/train sets 
#predictions = train(train_in,train_labels,test_in,test_labels,PLOT=True)

#extract predicted labels for entire dataset
output_labels = train(all_in,labels,items_in,item_labels,PLOT=True)
filename = 'trained_input.pickle'
with open(filename, 'wb') as f:
    pkl.dump(output_labels, f)

#concatenate for training SL model
full_set = np.concatenate([all_in,output_labels,all_out],axis=1)
np.random.shuffle(full_set)
training, test = full_set[:818,:], full_set[818:,:]
train_in = training[:,:6]
train_out = training[:,6:]
test_in = test[:,:6]
test_out = test[:,6:]

#train model using entire dataset (split into train and test to evaluate)
training_preds = model(train_in,train_out,test_in,test_out)

##############################################################################
    ######################### Simulation ###########################

all_in = np.concatenate([all_in,output_labels],axis=1)
items_in = np.concatenate([items_in,item_labels],axis=1)

#extract test_mse from last epoch training
testing_preds = model(all_in,all_out,items_in,items_out)

#evaluate accuracy by test category 
test_mse = (np.square(testing_preds - items_out)).mean(axis=1)
error_by_cat = np.array([test_mse[0:4],\
                                 test_mse[4:8],\
                                 test_mse[8:12],\
                                 test_mse[12:16]])

filename = 'trained_predictions.pickle'
with open(filename, 'wb') as f:
    pkl.dump(testing_preds, f)
filename = 'error_by_cat_trained.pickle'
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
fig.savefig('fig2.png', bbox_inches='tight')




