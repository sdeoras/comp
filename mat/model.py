import tensorflow as tf

# add a version to this graph
tf.constant("0.1.0", name="version")

# inputs:
buff_1 = tf.placeholder(dtype=tf.float64, name="buff1")
shape_1 = tf.placeholder(dtype=tf.int64, name="shape1")

buff_2 = tf.placeholder(dtype=tf.float64, name="buff2")
shape_2 = tf.placeholder(dtype=tf.int64, name="shape2")

shape_b = tf.placeholder(dtype=tf.int64, name="shapeBegin")
shape_s = tf.placeholder(dtype=tf.int64, name="shapeSize")

# inverse
inv = tf.reshape(buff_1, shape=shape_1)
inv = tf.linalg.inv(inv)
inv = tf.reshape(inv, shape=[-1], name="inv")

# transpose
transpose = tf.reshape(buff_1, shape=shape_1)
transpose = tf.linalg.transpose(transpose)
transpose = tf.reshape(transpose, shape=[-1], name="transposeOp")

# qr decomposition
qr = tf.reshape(buff_1, shape=shape_1)
q, r = tf.linalg.qr(qr, full_matrices=True)
q = tf.reshape(q, shape=[-1])
r = tf.reshape(r, shape=[-1])
tf.identity(q, name="qrdecomp_q")
tf.identity(r, name="qrdecomp_r")

# create matrix
mat_zeros = tf.zeros(shape=shape_1, dtype=tf.float64)
tf.reshape(mat_zeros, shape=[-1], name="zeros")

mat_ones = tf.ones(shape=shape_1, dtype=tf.float64)
tf.reshape(mat_ones, shape=[-1], name="ones")

mat_rand = tf.random_uniform(shape=shape_1, dtype=tf.float64)
tf.reshape(mat_rand, shape=[-1], name="rand")

mat_randn = tf.random_normal(shape=shape_1, dtype=tf.float64)
tf.reshape(mat_rand, shape=[-1], name="randn")

# matrix multiply
mat_x = tf.reshape(buff_1, shape=shape_1)
mat_y = tf.reshape(buff_2, shape=shape_2)
mat_xy = tf.matmul(mat_x, mat_y)
tf.shape(mat_xy, out_type=tf.int64, name="mulShape")
tf.reshape(mat_xy, shape=[-1], name="mul")

# matrix slice
mat_s = tf.reshape(buff_1, shape=shape_1)
mat_s = tf.slice(mat_s, begin=shape_b, size=shape_s)
mat_s_shape = tf.shape(mat_s, out_type=tf.int64)
tf.identity(mat_s_shape, name="sliceShapeOp")
mat_s =tf.reshape(mat_s, shape=[-1])
tf.identity(mat_s, name="sliceOp")

# matrix reshape
mat_reshape = tf.reshape(buff_1, shape=shape_1)
mat_reshape = tf.reshape(mat_reshape, shape=shape_2)
mat_reshape = tf.reshape(mat_reshape, shape=[-1])
tf.identity(mat_reshape, name="reshapeOp")

# matrix repeat aka tile
mat_tiled = tf.reshape(buff_1, shape=shape_1)
mat_tiled = tf.tile(mat_tiled, multiples=shape_2)
mat_tiled_shape = tf.shape(mat_tiled, out_type=tf.int64)
tf.identity(mat_tiled_shape, name="tileShapeOp")
mat_tiled =tf.reshape(mat_tiled, shape=[-1])
tf.identity(mat_tiled, name="tileOp")

# finally save the graph to be used in Go code
graph = tf.Session().graph_def
tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)

with tf.Session() as sess:
    tf.summary.FileWriter(logdir="/tmp/tensorflow/mat", graph=sess.graph)

print("run 'tensorboard --logdir=/tmp/tensorflow' to view the graph")
