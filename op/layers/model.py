import tensorflow as tf

# add a version to this graph
tf.constant("0.1.0", name="version")

# inputs:
my_input = tf.placeholder(dtype=tf.uint8, shape=[1, None, None, 3], name="myInput")
my_input_slice = tf.placeholder(dtype=tf.uint8, shape=[1, None, None], name="myInputSlice")

# operator:
# https://github.com/tensorflow/tensorflow/issues/23931
f = tf.layers.flatten(my_input)
tf.identity(f, name="flattenImage")

f2 = tf.layers.flatten(my_input_slice)
tf.identity(f2, name="flattenSlice")

# finally save the graph to be used in Go code
graph = tf.Session().graph_def
tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)

with tf.Session() as sess:
	tf.summary.FileWriter(logdir="/tmp/tensorflow/op/layers", graph=sess.graph)

print("run 'tensorboard --logdir=/tmp/tensorflow' to view the graph")
