import tensorflow as tf

# here we define 3 operations in the graph as follows
# 1) inputPath --> files
# 2) file --> buffer
# 3) file, buffer --> write somewhere

# note that both operations are independent paths in the graph with no
# common nodes or edges!

# graph operation #1 can be expressed as follows where
# "files" are matched files
input_dir = tf.placeholder(tf.string, name="inputPath")
tf.matching_files(input_dir, name="matchingFiles")

# graph operation #2 can be expressed as follows, where
# output of "readFile" operation is file content as string
# tensor
input_file = tf.placeholder(tf.string, name="inputFileName")
tf.read_file(input_file, name="readFile")

# graph operation #3 can be expressed as follows, where
# buffer can be written to an output file.
output_file = tf.placeholder(tf.string, name="outputFileName")
buffer = tf.placeholder(tf.string, name="outputFileBuffer")
tf.write_file(output_file, buffer, "writeFile")

# add a version to this graph
tf.constant("0.1.0", name="version")

# finally save the graph to be used in Go code
graph = tf.Session().graph_def
tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)

with tf.Session() as sess:
	tf.summary.FileWriter(logdir="/tmp/tensorflow/cloud", graph=sess.graph)

print("run 'tensorboard --logdir=/tmp/tensorflow' to view the graph")
