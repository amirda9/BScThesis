# lets just build an attention from the scratch
import tensorflow as tf
from keras_preprocessing import sequence


vocab_size = 10000

pad_id = 0
start_id = 1
oov_id = 2
index_offset = 2

tf.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)