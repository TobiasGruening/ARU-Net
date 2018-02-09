from __future__ import print_function, division

import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import misc
from pix_lab.util.util import load_graph

class Inference_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, img_list, scale=0.33, mode='L'):
        self.graph = load_graph(path_to_pb)
        self.img_list = img_list
        self.scale = scale
        self.mode = mode

    def inference(self, print_result=True, gpu_device="0"):
        val_size = len(self.img_list)
        if val_size is None:
            print("No Inference Data available. Skip Inference.")
            return
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            print("Start Inference")
            timeSum = 0.0
            for step in range(0, val_size):
                aTime = time.time()
                aImgPath = self.img_list[step]
                print(
                    "Image: {:} ".format(aImgPath))
                batch_x = self.load_img(aImgPath, self.scale, self.mode)
                print(
                    "Resolution: h {:}, w {:} ".format(batch_x.shape[1],batch_x.shape[2]))
                # Run validation
                aPred = sess.run(predictor,
                                       feed_dict={x: batch_x})
                curTime = (time.time() - aTime)*1000.0
                timeSum += curTime
                print(
                    "Update time: {:.2f} ms".format(curTime))
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
                            # misc.imsave('out' + str(aI) + '.jpg', aPred[0,:, :,aI-1])
                            a.set_title('Channel: ' + str(aI-1))
                    print('To go on just CLOSE the current plot.')
                    plt.show()
            self.output_epoch_stats_val(timeSum/val_size)

            print("Inference Finished!")

            return None

    def output_epoch_stats_val(self, time_used):
        print(
            "Inference avg update time: {:.2f} ms".format(time_used))

    def load_img(self, path, scale, mode):
        aImg = misc.imread(path, mode=mode)
        sImg = misc.imresize(aImg, scale, interp='bicubic')
        fImg = sImg
        if len(sImg.shape) == 2:
            fImg = np.expand_dims(fImg,2)
        fImg = np.expand_dims(fImg,0)

        return fImg