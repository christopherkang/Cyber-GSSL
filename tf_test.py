import tensorflow as tf
test = tf.constant([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
out = tf.reshape(test, [-1])
out2 = tf.reshape(out, [9, -1])

with tf.Session() as sess:
    print(sess.run((test, out, out2)))
    