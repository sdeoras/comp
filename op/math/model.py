import tensorflow as tf

# add a version to this graph
tf.constant("0.1.0", name="version")

# unique operator:
my_input = tf.placeholder(dtype=tf.float64, name="myInput")

# stats
tf.reduce_mean(my_input, name="mean")
tf.reduce_max(my_input, name="max")
tf.reduce_min(my_input, name="min")
tf.reduce_prod(my_input, name="prod")
tf.reduce_sum(my_input, name="sum")

# moments operator
mean, variance = tf.nn.moments(my_input, axes=[0])
tf.identity(variance, name="var")

# finally save the graph to be used in Go code
graph = tf.Session().graph_def
tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)

with tf.Session() as sess:
	tf.summary.FileWriter(logdir="/tmp/tensorflow/op/math", graph=sess.graph)

print("run 'tensorboard --logdir=/tmp/tensorflow' to view the graph")
