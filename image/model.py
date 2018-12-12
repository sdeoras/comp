import tensorflow as tf

# add a version to this graph
tf.constant("0.1.0", name="version")

# inputs:
file_name = tf.placeholder(dtype=tf.string, name="inputFileName")
input_size = tf.placeholder(dtype=tf.int32, name="inputSize")
input_buffer = tf.placeholder(dtype=tf.string, name="inputBuffer")
input_image = tf.placeholder(dtype=tf.uint8, name="inputImage")
input_raw = tf.placeholder(dtype=tf.float32, name="inputRaw")
input_mean = tf.placeholder(dtype=tf.float32, name="inputMean")
input_std = tf.placeholder(dtype=tf.float32, name="inputStd")
input_raw_images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name="inputRawImages")
input_images = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name="inputImages")

# encode and decode
image = tf.read_file(file_name)
tf.image.decode_jpeg(image, name="decodeFile")
tf.image.decode_jpeg(input_buffer, name="decode")
tf.image.encode_jpeg(input_image, name="encodeJpg")

# convert to grayscale
gray_images = tf.image.rgb_to_grayscale(input_images, name="rgbToGrayscale")

# sobel edge detection
output_sobel = tf.image.sobel_edges(input_raw)
tf.identity(output_sobel, name="sobel")

# resize
resized_image = tf.image.resize_bilinear(input_image, input_size, align_corners=True, name="resize")
normalized_image = tf.divide(tf.subtract(resized_image, [input_mean]), [input_std], name="resizeNormalize")

# gray resized, sliced and flattened
gray_resized = tf.image.resize_bilinear(gray_images, input_size, align_corners=True)
gray_normalized = tf.map_fn(lambda image: tf.image.per_image_standardization(image), gray_resized)
bw = gray_normalized[:, :, :, 0]
bw_flattened = tf.layers.flatten(bw)
meanBw, varianceBw = tf.nn.moments(bw_flattened, axes=[1], keep_dims=True)
tf.identity(bw_flattened, name="batch")

# moments
flat = tf.layers.flatten(input_raw_images)
mean, variance = tf.nn.moments(flat, axes=[1])
tf.identity(variance, name="var")
tf.identity(mean, name="mean")


# finally save the graph to be used in Go code
graph = tf.Session().graph_def
tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)

with tf.Session() as sess:
    tf.summary.FileWriter(logdir="/tmp/tensorflow/image", graph=sess.graph)

print("run 'tensorboard --logdir=/tmp/tensorflow' to view the graph")
