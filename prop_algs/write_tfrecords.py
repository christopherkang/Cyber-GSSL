import tensorflow as tf
import pandas as pd
import numpy as np

import os

os.chdir(os.path.dirname(__file__))

EDGE_MATRIX = pd.read_pickle("../data/pandas_weight_array.pickle")

EDGE_MATRIX = EDGE_MATRIX.head(10)

feature_dict = {str(key): tf.train.Feature(
    float_list=tf.train.FloatList(value=np.array(value)))
    for key, value in dict(EDGE_MATRIX).items()}
example = tf.train.Example(features=tf.train.Features(
    feature=feature_dict
))

# Write TFrecord file
with tf.python_io.TFRecordWriter('testweights.tfrecord') as writer:
    writer.write(example.SerializeToString())

"""
# Read and print data:
sess = tf.Session()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['testweights.tfrecord'])

_, serialized_example = reader.read(filename_queue)

# Define features
read_features = {str(key): tf.VarLenFeature(dtype=tf.float32) 
                 for key in np.arange(1,501)}

# Extract features from serialized data
read_data = tf.parse_single_example(serialized=serialized_example,
                                    features=read_features)

# Many tf.train functions use tf.train.QueueRunner,
# so we need to start it before we read
tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_data.items():
    print('{}: {}'.format(name, tensor.eval(session=sess)))
"""

# Even though TF Records works, it is too poorly documented to be reliable
# There is also disagreement as to whether this relies upon queue runners 


with tf.Session() as sess:
    dataset = tf.data.TFRecordDataset('testweights.tfrecord')
    dataset = dataset.map(lambda x: tf.decode_raw(x, out_type=tf.float32))
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator().get_next()
    print(iterator.eval())
    print(iterator.eval())
