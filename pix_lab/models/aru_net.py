from __future__ import print_function, division

import tensorflow as tf
from pix_lab.util import layers
from collections import OrderedDict
import logging

def attCNN(input, channels, activation):
    """
    Attention network
    :param input:
    :param channels:
    :param activation:
    :return:
    """
    with tf.variable_scope('attPart') as scope:
        conv1 = layers.conv2d_bn_lrn_drop('conv1', input, [4, 4, channels, 12], activation=activation)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        conv2 = layers.conv2d_bn_lrn_drop('conv2', pool1, [4, 4, 12, 16], activation=activation)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        conv3 = layers.conv2d_bn_lrn_drop('conv3', pool2, [4, 4, 16, 32], activation=activation)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        out_DS = layers.conv2d_bn_lrn_drop('conv4', pool3, [4, 4, 32, 1], activation=activation)
    return out_DS

def detCNN(input, useResidual, useLSTM, channels, scale_space_num, res_depth, featRoot,
           filter_size, pool_size, activation):
    """
    Feature Detection Network
    :param input:
    :param useResidual:
    :param useLSTM:
    :param channels:
    :param scale_space_num:
    :param res_depth:
    :param featRoot:
    :param filter_size:
    :param pool_size:
    :param activation:
    :return:
    """
    unetInp = input
    ksizePool = [1, pool_size, pool_size, 1]
    stridePool = ksizePool
    lastFeatNum = channels
    actFeatNum = featRoot
    dw_h_convs = OrderedDict()
    for layer in range(0, scale_space_num):
        with tf.variable_scope('unet_down_' + str(layer)) as scope:
            if useResidual:
                x = layers.conv2d_bn_lrn_drop('conv1', unetInp, [filter_size, filter_size, lastFeatNum, actFeatNum],
                                              activation=tf.identity)
                orig_x = x
                x = tf.nn.relu(x, name='activation')
                for aRes in range(0,res_depth):
                    if aRes < res_depth-1:
                        x = layers.conv2d_bn_lrn_drop('convR_' + str(aRes), x, [filter_size, filter_size, actFeatNum,
                                    actFeatNum], activation=activation)
                    else:
                        x = layers.conv2d_bn_lrn_drop('convR_' + str(aRes), x, [filter_size, filter_size, actFeatNum,
                                    actFeatNum], activation=tf.identity)
                x += orig_x
                x = activation(x, name='activation')
                dw_h_convs[layer] = x
            else:
                conv1 = layers.conv2d_bn_lrn_drop('conv1', unetInp, [filter_size, filter_size, lastFeatNum, actFeatNum],
                                                  activation=activation)
                dw_h_convs[layer] = layers.conv2d_bn_lrn_drop('conv2', conv1, [filter_size, filter_size, actFeatNum,
                                    actFeatNum], activation=activation)
            if layer < scale_space_num - 1:
                unetInp = tf.nn.max_pool(dw_h_convs[layer], ksizePool, stridePool, padding='SAME', name='pool')
            else:
                unetInp = dw_h_convs[layer]
            lastFeatNum=actFeatNum
            actFeatNum *= pool_size
    actFeatNum = lastFeatNum/pool_size
    if useLSTM:
        # Run separable 2D LSTM
        unetInp = layers.separable_rnn(unetInp, lastFeatNum, scope="RNN2D", cellType='LSTM')
    for layer in range(scale_space_num - 2, -1, -1):
        with tf.variable_scope('unet_up_' + str(layer)) as scope:
            # Upsampling followed by two ConvLayers
            dw_h_conv = dw_h_convs[layer]
            out_shape = tf.shape(dw_h_conv)
            deconv = layers.deconv2d_bn_lrn_drop('deconv', unetInp, [filter_size, filter_size, actFeatNum,lastFeatNum],
                            out_shape, pool_size, activation=activation)
            conc = tf.concat([dw_h_conv, deconv], 3, name='concat')
            if useResidual:
                x = layers.conv2d_bn_lrn_drop('conv1', conc, [filter_size, filter_size, pool_size*actFeatNum,
                                    actFeatNum], activation=tf.identity)
                orig_x = x
                x = tf.nn.relu(x, name='activation')
                for aRes in range(0,res_depth):
                    if aRes < res_depth-1:
                        x = layers.conv2d_bn_lrn_drop('convR_' + str(aRes), x, [filter_size, filter_size, actFeatNum,
                                    actFeatNum], activation=activation)
                    else:
                        x = layers.conv2d_bn_lrn_drop('convR_' + str(aRes), x, [filter_size, filter_size, actFeatNum,
                                    actFeatNum], activation=tf.identity)
                x += orig_x
                unetInp = activation(x, name='activation')
            else:
                conv1 = layers.conv2d_bn_lrn_drop('conv1', conc, [filter_size, filter_size, pool_size * actFeatNum,
                                    actFeatNum], activation=activation)
                unetInp = layers.conv2d_bn_lrn_drop('conv2', conv1, [filter_size, filter_size, actFeatNum, actFeatNum],
                                                activation=activation)
            lastFeatNum=actFeatNum
            actFeatNum /= pool_size
    return unetInp

def create_aru_net(inp, channels, n_class, scale_space_num, res_depth,
            featRoot, filter_size, pool_size, activation, model, num_scales):
    """
    Creates a neural pixel labeler of specified type. This NPL can process images of arbitrarily sizes

    :param inp: input tensor, shape [?,?,?,channels]
    :param channels: number of channels of the input image
    :param n_class: number of output labels
    :param scale_space_num: number of scale spaces
    :param res_depth: depth of residual blocks
    :param featRoot: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param model: what network type is used [u, ru, aru and laru]
    :param num_scales: number of scales for the scale space pyramid for the attention model
    """

    inp = tf.map_fn(lambda image: tf.image.per_image_standardization(image), inp)

    img_shape = tf.shape(inp)
    # Shape of the upsampled tensor
    o_shape = tf.stack([
        img_shape[0],
        img_shape[1],
        img_shape[2],
        featRoot
    ])

    useResidual = False
    useAttention = False
    useLSTM = False

    if 'ru' in model:
        useResidual = True
    if 'aru' in model:
        useAttention = True
    if 'laru' in model:
        useLSTM = True


    #Det Feature Maps
    out_det_map = OrderedDict()
    inp_scale_map = OrderedDict()
    inp_scale_map[0] = inp
    if useAttention:
        for sc in range(1, num_scales):
            inp_scale_map[sc] = tf.nn.avg_pool(inp_scale_map[sc-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('featMapG') as scope:
        out_0 = detCNN(inp,useResidual, useLSTM, channels, scale_space_num, res_depth,
                       featRoot, filter_size, pool_size, activation)
        out_det_map[0] = out_0
        if useAttention:
            scope.reuse_variables()
            upSc = 1
            for sc in range(1, num_scales):
                out_S = detCNN(inp_scale_map[sc], useResidual, useLSTM, channels, scale_space_num, res_depth,
                               featRoot, filter_size, pool_size, activation)
                upSc = upSc * 2
                out = layers.upsample_simple(out_S, o_shape, upSc, featRoot)
                out_det_map[sc] = out

    if useAttention:
        # Pay Attention
        out_att_map = OrderedDict()
        with tf.variable_scope('attMapG') as scope:
            upSc = 8
            for sc in range(0, num_scales):
                outAtt_O = attCNN(inp_scale_map[sc], channels, activation)
                outAtt_U = layers.upsample_simple(outAtt_O, tf.shape(inp), upSc, 1)
                scope.reuse_variables()
                out_att_map[sc] = outAtt_U
                upSc = upSc * 2
        val = []
        for sc in range(0, num_scales):
            val.append(out_att_map[sc])
        allAtt = tf.concat(values=val, axis=3)

        allAttSoftMax = tf.nn.softmax(allAtt)
        listOfAtt = tf.split(allAttSoftMax, num_scales, axis=3)
        val = []
        for sc in range(0, num_scales):
            val.append(tf.multiply(out_det_map[sc], listOfAtt[sc]))
        map = tf.add_n(val)
    else:
        map = out_det_map[0]
    logits = layers.conv2d_bn_lrn_drop('class', map, [4, 4, featRoot, n_class], activation=tf.identity)
    return logits


class ARUnet(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=1, n_class=2, model_kwargs={}):
        tf.reset_default_graph()

        self.n_class = n_class
        self.channels = channels

        self.x = tf.placeholder("float", shape=[None, None, None, self.channels], name="inImg")
        # These are not used now
        # self.keep_prob = tf.placeholder_with_default(1.0, [])
        # self.is_training = tf.placeholder_with_default(tf.constant(False), [])

        self.scale_space_num = model_kwargs.get("scale_space_num", 6)
        self.res_depth = model_kwargs.get("res_depth", 3)
        self.featRoot = model_kwargs.get("featRoot", 8)
        self.filter_size = model_kwargs.get("filter_size", 3)
        self.pool_size = model_kwargs.get("pool_size", 2)
        self.activation_name = model_kwargs.get("activation_name", "relu")
        if self.activation_name is "relu":
            self.activation = tf.nn.relu
        if self.activation_name is "elu":
            self.activation = tf.nn.elu
        self.model = model_kwargs.get("model", "aru")
        self.num_scales = model_kwargs.get("num_scales", 5)
        self.final_act = model_kwargs.get("final_act", "softmax")
        print("Model Type: " + self.model)
        logits = create_aru_net(self.x, self.channels, self.n_class, self.scale_space_num, self.res_depth,
                                self.featRoot, self.filter_size, self.pool_size, self.activation, self.model,
                                self.num_scales)
        self.logits = tf.identity(logits, 'logits')
        if self.final_act is "softmax":
            self.predictor = tf.nn.softmax(self.logits, name='output')
        elif self.final_act is "sigmoid":
            self.predictor = tf.nn.sigmoid(self.logits, name='output')
        elif self.final_act is "identity":
            self.predictor = tf.identity(self.logits, name='output')


    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver(max_to_keep=50)
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path, var_dict=None):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        if var_dict is None:
            saver = tf.train.Saver()
        else:
            saver = tf.train.Saver(var_list=var_dict)
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
