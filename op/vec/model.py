import tensorflow as tf

# add a version to this graph
tf.constant("0.1.0", name="version")

# linspace operator:
# here we define a graph that performs linspace operation given
# start, stop and num
start = tf.placeholder(tf.float64, name="start")
stop = tf.placeholder(tf.float64, name="stop")
num = tf.placeholder(tf.int64, name="num")
tf.linspace(start=start, stop=stop, num=num, name="linspace")

# cumulative sum operator:
my_input = tf.placeholder(tf.float64, name="myInput")
tf.math.cumsum(my_input, name="cumsum")
tf.math.cumprod(my_input, name="cumprod")
tf.reverse

# finally save the graph to be used in Go code
graph = tf.Session().graph_def
tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)

with tf.Session() as sess:
	tf.summary.FileWriter(logdir="/tmp/tensorflow/op/vec", graph=sess.graph)

print("run 'tensorboard --logdir=/tmp/tensorflow' to view the graph")
