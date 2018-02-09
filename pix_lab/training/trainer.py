from __future__ import print_function, division

import os
import time

import tensorflow as tf

from cost import get_cost
from optimizer import get_optimizer

class Trainer(object):
    """
    Trains a unet instance

    :param net: the arunet instance to train
    :param opt_kwargs: (optional) kwargs passed to the optimizer
    :param cost_kwargs: (optional) kwargs passed to the cost function

    """

    def __init__(self, net, opt_kwargs={}, cost_kwargs={}):
        self.net = net
        self.tgt = tf.placeholder("float", shape=[None, None, None, self.net.n_class])
        self.global_step = tf.placeholder(tf.int64)
        self.opt_kwargs = opt_kwargs
        self.cost_kwargs = cost_kwargs
        self.cost_type=cost_kwargs.get("cost_name", "cross_entropy")

    def _initialize(self, batch_steps_per_epoch, output_path):
        self.cost = get_cost(self.net.logits, self.tgt, self.cost_kwargs)
        self.optimizer, self.ema, self.learning_rate_node = get_optimizer(self.cost, self.global_step, batch_steps_per_epoch, self.opt_kwargs)

        init = tf.global_variables_initializer()
        if not output_path is None:
            output_path = os.path.abspath(output_path)
            if not os.path.exists(output_path):
                print("Allocating '{:}'".format(output_path))
                os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, restore_file=None, batch_steps_per_epoch=1024, epochs=250, gpu_device="0", max_spat_dim=5000000):
        """
        Launches the training process
        :param data_provider:
        :param output_path:
        :param restore_path:
        :param batch_size:
        :param batch_steps_per_epoch:
        :param epochs:
        :param keep_prob:
        :param gpu_device:
        :param max_spat_dim:
        :return:
        """
        print("Epochs: " + str(epochs))
        print("Batch Size Train: " + str(data_provider.batchsize_tr))
        print("Batchsteps per Epoch: " + str(batch_steps_per_epoch))
        if not output_path is None:
            save_path = os.path.join(output_path, "model")
        if epochs == 0:
            return save_path

        init = self._initialize(batch_steps_per_epoch, output_path)

        val_size = data_provider.size_val

        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.Session(config=session_conf) as sess:
            sess.run(init)

            if restore_file != None:
                print("Loading Checkpoint.")
                self.net.restore(sess, restore_file)
            else:
                print("Starting from scratch.")

            print("Start optimization")

            bestLoss = 100000.0
            shown_samples = 0
            for epoch in range(epochs):
                total_loss = 0
                lr = 0
                time_step_train = time.time()
                for step in range(( epoch * batch_steps_per_epoch), (( epoch + 1 ) * batch_steps_per_epoch)):
                    batch_x, batch_tgt = data_provider.next_data('train')
                    skipped = 0
                    if batch_x is None:
                        print("No Training Data available. Skip Training Path.")
                        break
                    while batch_x.shape[1] * batch_x.shape[2] > max_spat_dim:
                        batch_x, batch_tgt = data_provider.next_data('train')
                        skipped = skipped + 1
                        if skipped > 100:
                            print("Spatial Dimension of Training Data to high. Aborting.")
                            return save_path
                    # Run training
                    _, loss, lr = sess.run \
                        ([self.optimizer, self.cost, self.learning_rate_node],
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.tgt: batch_tgt,
                                                                 self.global_step: step})
                    shown_samples = shown_samples + batch_x.shape[0]
                    if self.cost_type is "cross_entropy_sum":
                        sh = batch_x.shape
                        loss /= sh[1] * sh[2] * sh[0]
                    total_loss += loss
                total_loss = total_loss/batch_steps_per_epoch
                time_used= time.time() - time_step_train
                self.output_epoch_stats_train(epoch+1, total_loss, shown_samples, lr, time_used)
                total_loss = 0
                time_step_val = time.time()
                for step in range(0,val_size):
                    batch_x, batch_tgt = data_provider.next_data('val')
                    if batch_x is None:
                        print("No Validation Data available. Skip Validation Path.")
                        break
                    # Run validation
                    loss, aPred = sess.run([self.cost, self.net.predictor], feed_dict={self.net.x: batch_x,
                                                                                       self.tgt: batch_tgt})
                    if self.cost_type is "cross_entropy_sum":
                        sh = batch_x.shape
                        loss /= sh[1] * sh[2] * sh[0]
                    total_loss += loss
                if val_size != 0:
                    total_loss = total_loss / val_size
                    time_used = time.time() - time_step_val
                    self.output_epoch_stats_val(epoch+1, total_loss, time_used)
                    data_provider.restart_val_runner()

                if not output_path is None:
                    if total_loss < bestLoss or (epoch + 1) % 10 == 0 :
                        if total_loss < bestLoss:
                            bestLoss = total_loss
                        save_pathAct = save_path + str(epoch + 1)
                        self.net.save(sess, save_pathAct)

            data_provider.stop_all()
            print("Optimization Finished!")
            print("Best Val Loss: " + str(bestLoss))
            return save_path


    def output_epoch_stats_train(self, epoch, total_loss, shown_sample, lr, time_used):
        print(
            "TRAIN: Epoch {:}, Average loss: {:.8f}, training samples shown: {:}, learning rate: {:.6f}, time used: {:.2f}".format(epoch, total_loss, shown_sample, lr, time_used))

    def output_epoch_stats_val(self, epoch, total_loss, time_used):
        print(
            "VAL: Epoch {:}, Average loss: {:.8f}, time used: {:.2f}".format(epoch, total_loss, time_used))

