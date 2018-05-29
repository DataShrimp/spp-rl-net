from keras.datasets import mnist
from keras.models import load_model
from SpatialPyramidPooling import SpatialPyramidPooling
import numpy as np
import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == "__mail__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    i = 0
    x_pred = x_test[i]
    np.savetxt("foo.csv", x_pred, delimiter=",")
    print("label:", y_test[i])

    model = load_model("spp-net.h5", custom_objects={"SpatialPyramidPooling":SpatialPyramidPooling})
    # original size
    #x = x_pred[np.newaxis, :]
    # changing size
    x = x_pred[np.newaxis, 5:19, 5:19]
    print("shape:", x.shape)
    y_pred = model.predict(x)
    print("predict:", y_pred)

if __name__ == "__test__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    i = 0
    x_pred = x = x_test[i][np.newaxis, 5:19, 5:19]

    graph = load_graph("spp-net.pb")
    for op in graph.get_operations():
        print(op.values())

    # will lose input&output op
    x = graph.get_tensor_by_name('prefix/conv2d_1/input:0')
    y = graph.get_tensor_by_name('prefix/dense_2:0')

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={x: x_pred})
        print(y_out)
