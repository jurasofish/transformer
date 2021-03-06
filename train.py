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

from modules import *
from pprint import pprint
import random, math
import matplotlib.pyplot as plt
from munch import *

def plot(sess, t):
    x, y = generate_x_y_data_v1(isTrain = False, batch_size = 1)
    preds = np.zeros_like(y)
    
    
    for j in range(model.y_seq_length):
        feed_dict = {model.x: x, model.y: preds}
        _preds = sess.run(model.pred, feed_dict)
        preds[0, j, 0] = _preds[0, j, 0]
    
    # pprint(y)
    # pprint(pred)
    
    fig, ax = plt.subplots()
    ax.plot(list(range(10)), y[0, :, 0], color='red')
    ax.plot(list(range(10)), preds[0, :, 0], color='green')
    fig.savefig('img/' + str(t) + "test.png")
    # plt.show()
    
    
    


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

        x_ = np.array([x1, np.zeros_like(x1), np.ones_like(x1), np.ones_like(x1)*6])
        y_ = np.array([y1, np.ones_like(x1)])
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
    


class Transformer():

    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        self.build_model()
        self.init_saver()
    
    def build_model(self):
        with tf.variable_scope('Transformer'):

        
            sample_x, sample_y = generate_x_y_data_v1(True, 3)
            # pprint(sample_x)
            # pprint(sample_y)
            x_seq_length = sample_x.shape[1]
            x_var_count = sample_x.shape[2]
            y_seq_length = len(sample_y[0, :])
            y_var_count = sample_y.shape[2]
            
            self.y_seq_length = y_seq_length
            
            # plot first example
            fig, ax = plt.subplots()
            ax.plot(list(range(x_seq_length)), sample_x[0, :, 0])
            ax.plot(list(range(x_seq_length, x_seq_length + y_seq_length)), sample_y[0, :, 0])
            # plt.show()
            
            print('x: length', x_seq_length, 'and each time step has', x_var_count, 'variables')
            print('y: length', y_seq_length, 'and each time step has', y_var_count, 'variables')
            # x is a multi-variate time series; y is a single-variate time series.
            self.x = tf.placeholder(tf.float32, shape=(None, x_seq_length, x_var_count))
            self.y = tf.placeholder(tf.float32, shape=(None, y_seq_length, y_var_count))
            
            # define decoder inputs
            # Remove the final element from every sequence in y, 
            # then add the value 0 to the beginning of every sentence.
            # in effect, shift the series one point to the right.
            self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1)
                       
            # Encoder
            with tf.variable_scope("encoder"):
                '''Use conv1d to embed each time step, as per paper "attend and diagnose" '''
                # print('before ff:', self.x)
                self.enc = embed_conv(inputs=self.x, num_units=config.tra.hidden_units, scope="enc_embed_conv")
                # print('after ff: ', self.enc)
                
                ## Positional Encoding
                if config.tra.sinusoid:
                    self.enc += positional_encoding(self.x,
                                      num_units=config.tra.hidden_units, 
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
                                      num_units=config.tra.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    
                
                ## Dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=config.tra.dropout_rate)
                
                
                ## Blocks
                '''
                Remember that self.enc is the output of the encoder,
                so adding more and more blocks on top of it is just 
                building the encoder.
                '''
                for i in range(config.tra.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=config.tra.hidden_units, 
                                                        num_heads=config.tra.num_heads, 
                                                        dropout_rate=config.tra.dropout_rate,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*config.tra.hidden_units, config.tra.hidden_units])
            
            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                # Same as input, essentially.
                '''Use conv1d to embed each time step, as per paper "attend and diagnose" '''
                self.dec = embed_conv(inputs=self.decoder_inputs, num_units=config.tra.hidden_units, scope="dec_embed_conv")
                
                ## Positional Encoding
                # Same as input, essentially.
                if config.tra.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=config.tra.maxlen, 
                                      num_units=config.tra.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=y_seq_length, 
                                      num_units=config.tra.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=config.tra.dropout_rate)
                
                
                ## Blocks
                for i in range(config.tra.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=config.tra.hidden_units, 
                                                        num_heads=config.tra.num_heads, 
                                                        dropout_rate=config.tra.dropout_rate,
                                                        causality=True, 
                                                        scope="self_attention")
                        
                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=config.tra.hidden_units, 
                                                        num_heads=config.tra.num_heads,
                                                        dropout_rate=config.tra.dropout_rate,
                                                        causality=False,
                                                        scope="vanilla_attention")
                        
                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4*config.tra.hidden_units, config.tra.hidden_units])
                
            # Final linear projection
            # Final dimension is vocabulary size of english.
            pprint(self.dec)
            self.pred = tf.layers.dense(self.dec, y_var_count)
            pprint(self.pred)
            self.loss = tf.reduce_sum(tf.square(self.pred - self.y))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss)
               
            # Summary 
            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()

            
    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            
    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        
        
        
if __name__ == '__main__':                
    
    config = {
    "max_to_keep": 10,
	"lr": 0.000075,
        "tra": {
            "hidden_units": 32,
            "num_blocks": 2,
            "num_heads": 2,
            "dropout_rate": 0.1,
            "sinusoid": False} }
    config = munchify(config)
    sess = tf.InteractiveSession()
    model = Transformer(config=config); print("Graph loaded")
    sess.run(tf.global_variables_initializer())


    sess.run(tf.global_variables_initializer())
    for t in range(100000): 
        if(t % 25 == 0):
            plot(sess, t)
        x, y = generate_x_y_data_v1(isTrain = False, batch_size = 100)
        # pprint(x)
        feed_dict = {model.x: x, model.y: y}
        _, loss = sess.run([model.train_op, model.loss], feed_dict)
        pprint(loss)

    
    print("Done")    
    

