import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from SpatialPyramidPooling import SpatialPyramidPooling

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    :param session: 需要转换的tensorflow的session
    :param keep_var_names:需要保留的variable，默认全部转换constant
    :param output_names:output的名字
    :param clear_devices:是否移除设备指令以获得更好的可移植性
    :return:
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        #for v in tf.global_variables():
        #    print(v.op.name)
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

if __name__ == "__main__":
    model = load_model("spp-net.h5", custom_objects={"SpatialPyramidPooling":SpatialPyramidPooling})
    print(model.input.op.name)
    print(model.output.op.name)

    frozen_graph = freeze_session(K.get_session())
    tf.train.write_graph(frozen_graph, "./", "spp-net.pb", as_text=False)
