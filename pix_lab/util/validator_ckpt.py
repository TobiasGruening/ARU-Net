from __future__ import print_function, division

import time

import matplotlib.pyplot as plt
import tensorflow as tf

from pix_lab.training.cost import get_cost

class Validator_ckpt(object):
    """
    Validate a arunet instance

    :param net: the arunet instance to train

    """

    def __init__(self, net, cost_kwargs={}):
        self.net = net
        self.tgt = tf.placeholder("float", shape=[None, None, None, self.net.n_class])
        self.cost_kwargs = cost_kwargs
        self.cost_type = cost_kwargs.get("cost_name", "cross_entropy")

    def validate(self, data_provider, restore_ckt_path=None, use_ema_vars=False, print_result=True, gpu_device="0"):
        self.cost = get_cost(self.net.logits, self.tgt, self.cost_kwargs)
        if use_ema_vars:
            ema = tf.train.ExponentialMovingAverage(decay=1.0)
        init = tf.global_variables_initializer()
        val_size = data_provider.size_val
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
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
                # else:
                #     print("Error. Unable to load Checkpoint.")
            else:
                print("Error. You have to provide a path to restore a checkpoint.")

            print("Start validation")

            total_loss = 0
            time_val_step = time.time()
            for step in range(0, val_size):
                batch_x, batch_tgt = data_provider.next_data('val')
                if batch_x is None:
                    print("No Validation Data available. Skip Validation Path.")
                    break
                # allAtt = tf.get_default_graph().get_tensor_by_name("allAttMaps:0")
                # Run validationloss
                loss, aPred = sess.run([self.cost, self.net.predictor],
                                       feed_dict={self.net.x: batch_x, self.tgt: batch_tgt})
                if self.cost_type is "cross_entropy_sum":
                    sh = batch_x.shape
                    loss /= sh[1] * sh[2] * sh[0]
                total_loss += loss
                print("Act Loss: " + str(loss))
                print("Avg Loss: " + str(total_loss/(step+1)))
                if print_result:
                    n_class = self.net.n_class
                    channels = self.net.channels
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

