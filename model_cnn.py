#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:58:10 2018

@author: shanmukha
"""
from dataset_extractor import dataset_maker
import tensorflow as tf
import numpy as np

epochs = 60
l_rate = 0.001
dropout = 0.2
nesterov_momentum = 0.9
batch_size = 32
num_classes = 602
full_set,train_set,validation_set,test_set = dataset_maker()

full_x = [full_set[n][0] for n in range(len(full_set))]
full_y = [full_set[n][1] for n in range(len(full_set))]
train_x = [train_set[n][0] for n in range(len(train_set))]
train_y = [train_set[n][1] for n in range(len(train_set))]
val_x = [validation_set[n][0] for n in range(len(validation_set))]
val_y = [validation_set[n][1] for n in range(len(validation_set))]
test_x = [test_set[n][0] for n in range(len(test_set))]
test_y = [test_set[n][1] for n in range(len(test_set))]




img = tf.placeholder('float',[None,32,32,3])
class_index = tf.placeholder(tf.int64)

def cnn_model(img):
    img = tf.reshape(img,[-1,32,32,3])
    conv_l1 = tf.layers.conv2d(img,filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    conv_l2 = tf.layers.conv2d(conv_l1,filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    conv_l3 = tf.layers.conv2d(conv_l2,filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    
    pool_l1 = tf.layers.max_pooling2d(conv_l3,pool_size=2,strides=2,padding='same')
    
    conv_l4 = tf.layers.conv2d(pool_l1,filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    conv_l5 = tf.layers.conv2d(conv_l4,filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    conv_l6 = tf.layers.conv2d(conv_l5,filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    
    pool_l2 = tf.layers.max_pooling2d(conv_l6,pool_size=2,strides=2,padding='same')
    
    output = tf.reshape(pool_l2,[-1,8*8*64])
    
    fc_l1 = tf.contrib.layers.fully_connected(output,4096,activation_fn=None)
    fc_l1 = tf.layers.dropout(fc_l1,rate=dropout)
    fc_l2 = tf.contrib.layers.fully_connected(fc_l1,2048,activation_fn=None)
    fc_l2 = tf.layers.dropout(fc_l2,rate=dropout)
    
    result = tf.contrib.layers.fully_connected(fc_l2,num_classes,activation_fn=None)
    return result

def cnn_trainer(img):
    prediction = cnn_model(img)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=class_index,logits=prediction))
    optimizer = tf.train.MomentumOptimizer(l_rate,nesterov_momentum,use_nesterov=True).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print('starting epoch :',epoch)
            epoch_loss = 0
            start = 0
            end = batch_size
            for i in range(int(len(train_set)/batch_size)):
                epoch_x = []
                epoch_y = []
                epoch_x = [train_set[n][0] for n in range(start,end)]
                epoch_y = [train_set[n][1] for n in range(start,end)]
                #assert all(x.shape == (32,32,3) for x in epoch_x)
                j,c = sess.run([optimizer,loss],feed_dict={img:epoch_x,class_index:epoch_y})
                epoch_loss += c
                start = end
                end += batch_size
            print('completed epoch :',epoch)
            print('epoch loss :',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),class_index)
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        #print('Accuracy on validation set:',accuracy.eval({img:val_x,class_index:val_y}))
        print('Accuracy on test set:',accuracy.eval({img:test_x,class_index:test_y}))
cnn_trainer(img)
#print(np.array(train_set[1][1]).shape)        

    
    
    