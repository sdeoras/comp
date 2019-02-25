import tensorflow as tf

# add a version to this graph
tf.constant("0.1.0", name="version")

# configuration
# num_variables = tf.placeholder(dtype=tf.int32, shape=(), name="numVariables")
# num_classes = tf.placeholder(dtype=tf.int32, shape=(), name="numClasses")
learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name="learningRate")

# inputs and variables
x = tf.placeholder(tf.float32, shape=(None, None), name="x")
labels = tf.placeholder(tf.float32, shape=[None, None], name="labels")

# variables can be initialized using placeholder data.
# see more: https://github.com/tensorflow/tensorflow/issues/5972#issuecomment-265844831
w_input = tf.placeholder(dtype=tf.float32, shape=[None, None], name="weightsInit")
w = tf.Variable(w_input, validate_shape=False, name="weights")

b_input = tf.placeholder(dtype=tf.float32, shape=[None], name="biasesInit")
b = tf.Variable(b_input, validate_shape=False, name="biases")

# init op
init = tf.global_variables_initializer()

# model
y = tf.nn.softmax(tf.matmul(x, w) + b)

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
