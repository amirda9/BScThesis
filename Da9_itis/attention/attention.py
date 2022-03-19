# lets just build an attention from the scratch
from importlib.resources import path
import numpy as np
import tensorflow as tf
from keras_preprocessing import sequence
from tensorflow.keras.models import *
from tensorflow.keras.layers import *



vocab_size = 10000

pad_id = 0 
start_id = 1
oov_id = 2
index_offset = 2

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(path='imdb.npz',num_words=vocab_size,start_char=start_id,oov_char = oov_id, index_from=index_offset)

word2idx = tf.keras.datasets.imdb.get_word_index()

idx2word = {v+index_offset :k for k,v in word2idx.items()}
idx2word[pad_id] = '<PAD>'
idx2word[start_id] = '<START>'
idx2word[oov_id] = '<OOV>'


max_len = 200
rnn_cell_size = 128

x_train = sequence.pad_sequences(x_train,maxlen=max_len,truncating='post',padding='post',value=pad_id)
x_test = sequence.pad_sequences(x_train,maxlen=max_len,truncating='post',padding='post',value=pad_id)



# self attention 
model = Sequential()
model.add(Embedding(vocab_size,100,input_length=max_len))
model.add(Bidirectional(LSTM(units=16,dropout=0.5,return_sequences=True,recurrent_dropout=0.7)))
model.add(SeqSelfA)


model.summary()