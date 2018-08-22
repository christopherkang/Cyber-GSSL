import tensorflow as tf

alpha_1 = tf.constant(0.5, name="alpha_1")
alpha_2 = tf.constant(0.5, name="alpha_2")
alpha_3 = tf.constant(0.5, name="alpha_3")

w_uv = tf.placeholder(dtype=tf.float32)


tf.

def special_loss(weight, input_1, input_2):
    tf.matmul(tf.matmul(alpha_1, input_1), input_2)

# initialize variables 
start_var = tf.get_variable(initializer=tf.variables_initializer)