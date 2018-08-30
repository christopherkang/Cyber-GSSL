import tensorflow as tf
test = tf.constant([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
search = tf.where(tf.equal(0, test))
with tf.Session() as sess:
    print(sess.run((search)))
    