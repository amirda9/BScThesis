from unicodedata import bidirectional
import tensorflow as tf
import numpy as np
from keras_preprocessing import sequence
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from keras_self_attention import *
import plot

vocab_size = 10000

pad_id = 0
start_id = 1
oov_id = 2
index_offset = 0

(x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data(num_words=vocab_size,start_char=start_id,oov_char=oov_id,index_from=index_offset)

word2id = keras.datasets.imdb.get_word_index()

idx2word = {i:word for word,i in word2id.items()}

idx2word[pad_id] = '<PAD>'
idx2word[start_id] = '<START>'
# idx2word[oov_id] = '<OOV>'

max_len = 200
rnn_cell_size = 64

x_train = sequence.pad_sequences(x_train,maxlen=max_len,truncating = 'post',padding = 'post' , value=pad_id)
x_test = sequence.pad_sequences(x_test,maxlen=max_len,truncating = 'post',padding = 'post' , value=pad_id)


model = Sequential()
model.add(Embedding(vocab_size,100,input_length=max_len))
model.add(Bidirectional(LSTM(units=16,return_sequences=True,dropout=0.5,recurrent_dropout=0.5)))
# model.add(SeqSelfAttention(attention_activation='sigmoid',attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.summary()

# model = Sequential()
# model.add(Embedding(vocab_size,100,input_length=max_len))
# model.add(MultiHeadAttention(8,max_len))
# model.add(Flatten())
# model.add(Dense(1,activation='sigmoid'))
# model.summary()

sentence = []
inverted_word_index = dict((i, word) for (word, i) in word2id.items())
decoded_sequence = " ".join(idx2word[i] for i in x_train[80])
for i in x_train[80]:
    sentence.append(idx2word[i])
print(decoded_sequence,y_train[80])

print(sentence[89])
plot.setup_gui(sentence)


# with tf.device('/gpu:0'):
#     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#     history = model.fit(x_train,y_train,batch_size=128,epochs=2,validation_split=0.2, verbose=1)


#     res = model.evaluate(x_test,y_test)
#     print(res)
#     model.save('./NOATT_model.h5')