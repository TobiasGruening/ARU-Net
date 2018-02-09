from __future__ import print_function, division

import time

import matplotlib.pyplot as plt
import tensorflow as tf

from pix_lab.training.cost import get_cost
from pix_lab.util.util import load_graph

class Validator_pb(object):
    """
        Validate a arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, n_class, cost_kwargs={}):
        self.graph = load_graph(path_to_pb)
        self.cost_kwargs = cost_kwargs
        self.n_class=n_class

    def validate(self, data_provider, print_result=True, gpu_device="0"):
        val_size = data_provider.size_val
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            log = self.graph.get_tensor_by_name('logits:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            tgt = tf.placeholder("float", shape=[None, None, None, self.n_class])
            cost = get_cost(log, tgt, self.cost_kwargs)
            print("Start validation")

            total_loss = 0
            time_val_step = time.time()
            for step in range(0, val_size):
                batch_x, batch_tgt = data_provider.next_data('val')
                if batch_x is None:
                    print("No Validation Data available. Skip Validation Path.")
                    break
                # Run validation
                loss, aPred = sess.run([cost, predictor],
                                       feed_dict={x: batch_x, tgt: batch_tgt})
                total_loss += loss
                print("Act Loss: " + str(loss))
                if print_result:
                    n_class = aPred.shape[3]
                    channels = batch_x.shape[3]
                    fig = plt.figure()
                    for aI in range(0, n_class+1):
                        if aI == 0:
                            a = fig.add_subplot(1, n_class+1, 1)
                            if channels == 1:
                                plt.imshow(batch_x[0, :, :, 0], cmap=plt.cm.gray)
                            else:
                                plt.imshow(batch_x[0, :, :, :])
                            a.set_title('input')
                        else:
                            a = fig.add_subplot(1, n_class+1, aI+1)
                            plt.imshow(aPred[0,:, :,aI-1], cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
                            a.set_title('Channel: ' + str(aI-1))
                    plt.show()
            total_loss = total_loss / val_size
            time_used = time.time() - time_val_step
            self.output_epoch_stats_val(total_loss, time_used)

            data_provider.stop_all()
            print("Validation Finished!")

            return None

    def output_epoch_stats_val(self, total_loss, time_used):
        print(
            "VAL: Average loss: {:.8f}, time used: {:.2f}".format(total_loss, time_used))

