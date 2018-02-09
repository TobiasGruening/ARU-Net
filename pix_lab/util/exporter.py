from __future__ import print_function, division


import tensorflow as tf
from tensorflow.python.framework import graph_util


class Exporter_ckpt(object):
    """
    Export a arunet instance as pb file.

    :param net: the arunet instance to train

    """
    def __init__(self, net):
        self.net = net

    def export(self, restore_ckt_path=None, export_name=None, use_ema_vars=False, output_nodes=['output']):
        if use_ema_vars:
            ema = tf.train.ExponentialMovingAverage(decay=1.0)
        init = tf.global_variables_initializer()
        session_conf = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        with tf.Session(config=session_conf) as sess:
            sess.run(init)

            if restore_ckt_path != None:
                # ckpt = tf.train.get_checkpoint_state(restore_path)
                # if ckpt and ckpt.model_checkpoint_path:
                if use_ema_vars:
                    varTr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    varMapper = {}
                    for v in varTr:
                        shadow_v = ema.average_name(v)
                        varMapper[shadow_v] = v
                    print("Loading Checkpoint (Shadow Variables).")
                else:
                    varMapper=None
                    print("Loading Checkpoint.")
                self.net.restore(sess, restore_ckt_path, varMapper)
            else:
                print("Error. You have to provide a path to restore a checkpoint.")

            print("Export It")

            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()

            output_graph_def = graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                input_graph_def,  # The graph_def is used to retrieve the nodes
                output_nodes  # The output node names are used to select the usefull nodes
            )
            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(export_name, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

            print("Export Finished!")

            return None
