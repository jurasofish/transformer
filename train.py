# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from modules import *
import os, codecs
from tqdm import tqdm
from pprint import pprint
import random, math
import matplotlib.pyplot as plt


def generate_x_y_data_v1(isTrain, batch_size):
    """
    Data for exercise 1.
    returns: tuple (X, Y)
        X is a sine and a cosine from 0.0*pi to 1.5*pi
        Y is a sine and a cosine from 1.5*pi to 3.0*pi
    Therefore, Y follows X. There is also a random offset
    commonly applied to X an Y.
    The returned arrays are of shape:
        (seq_length, batch_size, output_dim)
        Therefore: (10, batch_size, 2)
    For this exercise, let's ignore the "isTrain"
    argument and test on the same data.
    """
    seq_length = 10

    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        rand = random.random() * 2 * math.pi

        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        x1 = sig1[:seq_length]
        y1 = sig1[seq_length:]

        x_ = np.array([x1, np.zeros_like(x1)])
        y_ = np.array([y1])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)
    
    '''
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)
    '''

    return batch_x, batch_y
    


class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():

        
            sample_x, sample_y = generate_x_y_data_v1(True, 3)
            pprint(sample_x)
            pprint(sample_y)
            x_seq_length = sample_x.shape[1]
            x_var_count = sample_x.shape[2]
            y_seq_length = len(sample_y[0, :])
            y_var_count = sample_y.shape[2]
            
            # plot first example
            fig, ax = plt.subplots()
            ax.plot(list(range(x_seq_length)), sample_x[0, :, 0])
            ax.plot(list(range(x_seq_length, x_seq_length + y_seq_length)), sample_y[0, :, 0])
            # plt.show()
            
            print('x: length', x_seq_length, 'and each time step has', x_var_count, 'variables')
            print('y: length', y_seq_length)
            # x is a multi-variate time series; y is a single-variate time series.
            self.x = tf.placeholder(tf.float32, shape=(None, sample_x.shape[1], sample_x.shape[0]))
            self.y = tf.placeholder(tf.float32, shape=(None, sample_y.shape[1]))
            
            # define decoder inputs
            # Remove the final element from every sequence in y, 
            # then add the value 0 to the beginning of every sentence.
            # in effect, shift the series one point to the right.
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*0, self.y[:, :-1]), -1)
                       
            # Encoder
            with tf.variable_scope("encoder"):
                '''Use conv1d to embed each time step, as per paper "attend and diagnose" '''
                print('before ff:', self.x)
                self.enc = embed_conv(inputs=self.x, num_units=hp.hidden_units, scope="enc_embed_conv")
                print('after ff: ', self.enc)
                
                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                else:
                    '''
                    tf.shape(self.x)[1] = max sentence length
                    tf.range(tf.shape(self.x)[1]) = [0, 1, 2, ..., max sentence length]
                    tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0) = [[0, 1, 2, ..., max sentence length]] (a matrix with one row and n columns)
                    tf.shape(self.x)[0] = batch size
                    tf.tile([0, 1, 2, ..., max sentence length]], [batch_size, 1])
                        = a 2D matrix, with batch_size rows. each row is [0, 1, 2, ..., max sentence length].
                        =   [[0 1 ... max_sent.]
                                    ...
                             [0 1 ... max_sent.]
                             [0 1 ... max_sent.]]
                    By putting this through the embedding function, every element
                    of the above matrix is replaced with a vector of length hp.hidden_units.
                    So, the ultimate effect is that every word (represented by a vector)
                    in self.enc has a vector added to it depending on it's position in
                    the sentence. This vector is trainable, and is always the same for the 
                    same position in the sentence.
                    '''
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=x_seq_length, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    
                '''
                ## Dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                '''
                
                ## Blocks
                '''
                Remember that self.enc is the output of the encoder,
                so adding more and more blocks on top of it is just 
                building the encoder.
                '''
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])
            
            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                # Same as input, essentially.
                self.dec = embedding(self.decoder_inputs, 
                                      vocab_size=len(en2idx), 
                                      num_units=hp.hidden_units,
                                      scale=True, 
                                      scope="dec_embed")
                
                ## Positional Encoding
                # Same as input, essentially.
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                '''
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                '''
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        
                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        
                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
                
            # Final linear projection
            # Final dimension is vocabulary size of english.
            self.logits = tf.layers.dense(self.dec, len(en2idx))
            # Select the most likely word from the final dimension
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            # tf.to_float(tf.not_equal([0, 1, 2, 3], 0)) = [0. 1. 1. 1.]
            # self.istarget is a mask?
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            # self.acc is a value from 0 to 1 of how many of the output words match y.
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
                
            if is_training:  
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()

if __name__ == '__main__':                
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")

    with tf.session(graph=g.graph) as sess:
        for t in range(10000): 
            feed_dict = {'blah': 'placeholder'}
            sess.run(g.train_op, feed_dict = feed_dict)

    
    print("Done")    
    

