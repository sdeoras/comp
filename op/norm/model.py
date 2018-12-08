import tensorflow as tf

# add a version to this graph
tf.constant("0.1.0", name="version")

# unique operator:
my_input = tf.placeholder(dtype=tf.float64, name="myInput")
tf.nn.softmax(my_input, name="softmax")

# finally save the graph to be used in Go code
graph = tf.Session().graph_def
tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)

with tf.Session() as sess:
    tf.summary.FileWriter(logdir="/tmp/tensorflow/op/norm", graph=sess.graph)

print("run 'tensorboard --logdir=/tmp/tensorflow' to view the graph")
