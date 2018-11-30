import tensorflow as tf

with tf.gfile.GFile('/Users/sdeoras/Downloads/graph.pb', "rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	g_in = tf.import_graph_def(graph_def, name="")

with tf.Session() as sess:
	tf.summary.FileWriter(logdir="/tmp/tensorflow/graph", graph=sess.graph)

print("tensorboard --logdir=/tmp/tensorflow")
