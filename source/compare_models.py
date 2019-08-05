#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:35:14 2018

@author: elizabethhutton
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import pickle as pkl

##############################################################################
    ######################### Model Comparison ###########################

filename = 'error_by_cat_trained.pickle'
with open(filename, 'rb') as f:
    error_trained = pkl.load(f)
filename = 'error_by_cat_simple.pickle'
with open(filename, 'rb') as f:
    error_simple = pkl.load(f)
    
###plot results
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()
bps = plt.boxplot(error_simple.transpose(), positions=np.array(range(len(error_simple)))*2.0-0.4, sym='', widths=0.6)
bpt = plt.boxplot(error_trained.transpose(), positions=np.array(range(len(error_trained)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bps, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpt, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Simple')
plt.plot([], c='#2C7BB6', label='Trained')
plt.legend()
test_cats = ('Word', 'Part Word', 'Cor. Traj.', 'Inc. Traj')
plt.xticks(range(0, len(test_cats) * 2, 2), test_cats)
plt.xlim(-2, len(test_cats)*2)
plt.ylim(-0.25, 2.5)
plt.ylabel('Error')
plt.title('Test Errors by Condition')
plt.tight_layout()
plt.savefig('boxcompare.png')    
    




