import tensorflow as tf
from tensorflow.python.framework import graph_util

output_node_names = "y_pred"
saver = tf.train.import_meta_graph('../trained_model/model.meta', clear_devices=True)

graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "../trained_model/model")
output_graph_def = graph_util.convert_variables_to_constants(
    sess,  # The session is used to retrieve the weights
    input_graph_def,  # The graph_def is used to retrieve the nodes
    output_node_names.split(",")  # The output node names are used to select the usefull nodes
)
output_graph = "../webapp/model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
