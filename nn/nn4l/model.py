import tensorflow as tf

# add a version to this graph
tf.constant("0.1.0", name="version")

# configuration
# num_variables = tf.placeholder(dtype=tf.int32, shape=(), name="numVariables")
# num_classes = tf.placeholder(dtype=tf.int32, shape=(), name="numClasses")
learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name="learningRate")

# inputs and variables
y_0 = tf.placeholder(tf.float32, shape=(None, None), name="x")
labels = tf.placeholder(tf.float32, shape=[None, None], name="labels")

# variables can be initialized using placeholder data.
# see more: https://github.com/tensorflow/tensorflow/issues/5972#issuecomment-265844831
w_input_layer_1 = tf.placeholder(dtype=tf.float32, shape=[None, None], name="weightsInitLayer1")
w_layer_1 = tf.Variable(w_input_layer_1, validate_shape=False, name="weightsLayer1")
b_input_layer_1 = tf.placeholder(dtype=tf.float32, shape=[None], name="biasesInitLayer1")
b_layer_1 = tf.Variable(b_input_layer_1, validate_shape=False, name="biasesLayer1")

w_input_layer_2 = tf.placeholder(dtype=tf.float32, shape=[None, None], name="weightsInitLayer2")
w_layer_2 = tf.Variable(w_input_layer_2, validate_shape=False, name="weightsLayer2")
b_input_layer_2 = tf.placeholder(dtype=tf.float32, shape=[None], name="biasesInitLayer2")
b_layer_2 = tf.Variable(b_input_layer_2, validate_shape=False, name="biasesLayer2")

w_input_layer_3 = tf.placeholder(dtype=tf.float32, shape=[None, None], name="weightsInitLayer3")
w_layer_3 = tf.Variable(w_input_layer_3, validate_shape=False, name="weightsLayer3")
b_input_layer_3 = tf.placeholder(dtype=tf.float32, shape=[None], name="biasesInitLayer3")
b_layer_3 = tf.Variable(b_input_layer_3, validate_shape=False, name="biasesLayer3")

w_input_layer_4 = tf.placeholder(dtype=tf.float32, shape=[None, None], name="weightsInitLayer4")
w_layer_4 = tf.Variable(w_input_layer_4, validate_shape=False, name="weightsLayer4")
b_input_layer_4 = tf.placeholder(dtype=tf.float32, shape=[None], name="biasesInitLayer4")
b_layer_4 = tf.Variable(b_input_layer_4, validate_shape=False, name="biasesLayer4")

# init op
init = tf.global_variables_initializer()

# 4 layer model
y_1 = tf.nn.relu(tf.matmul(y_0, w_layer_1) + b_layer_1)
y_2 = tf.nn.relu(tf.matmul(y_1, w_layer_2) + b_layer_2)
y_3 = tf.nn.relu(tf.matmul(y_2, w_layer_3) + b_layer_3)
y_4 = tf.nn.softmax(tf.matmul(y_3, w_layer_4) + b_layer_4)
y = y_4

# comparators
truth = tf.argmax(labels, 1, name="truth")
prediction = tf.argmax(y, 1, name="prediction")

# loss func
cross_entropy = -tf.reduce_sum(labels * tf.nn.log_softmax(y))
tf.identity(cross_entropy, name="crossEntropy")

# accuracy
is_correct = tf.equal(prediction, truth)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
tf.identity(accuracy, name="accuracy")

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# trainer
# execute this step as an end point feeding in three things:
# data, labels and learning rate
train_step = optimizer.minimize(cross_entropy, name="trainStep")

# summaries for tensorboard
# Create a summary to monitor cost tensor
cross_entropy_training_summary= tf.summary.scalar("training/cost", cross_entropy)
# Create a summary to monitor accuracy tensor for training data
accuracy_training = tf.summary.scalar("training/accuracy", accuracy)
# Create a summary to monitor accuracy tensor for validation data
# please note that this expression is exact copy of the one above
# since the model data flow (graph) is the same for both training
# and validation, the easy way to separate training and validation
# monitoring in tensorboard is by requesting accuracy values
# using different labels.
accuracy_validation = tf.summary.scalar("validation/accuracy", accuracy)

# merge training summaries
training_summary = tf.summary.merge([cross_entropy_training_summary, accuracy_training], collections=["training"])
tf.identity(training_summary, name="summaryTraining")

# merge validation summaries
training_summary = tf.summary.merge([accuracy_validation], collections=["validation"])
tf.identity(training_summary, name="summaryValidation")

# finally save the graph to be used in Go code
graph = tf.Session().graph_def
tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)

with tf.Session() as sess:
    tf.summary.FileWriter(logdir="/tmp/tensorflow/train", graph=sess.graph)

print("run 'tensorboard --logdir=/tmp/tensorflow' to view the graph")
