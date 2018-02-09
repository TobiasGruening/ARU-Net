import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell

def conv2d_bn_lrn_drop(scope_or_name,
                       inputs,
                       kernel_shape,
                       strides=[1, 1, 1, 1],
                       activation=tf.nn.relu,
                       use_bn=False,
                       use_mvn=False,
                       is_training=True,
                       use_lrn=False,
                       keep_prob=1.0,
                       dropout_maps=False,
                       initOpt=0,
                       biasInit=0.1):
    """Adds a 2-D convolutional layer given 4-D `inputs` and `kernel` with optional BatchNorm, LocalResponseNorm and Dropout.

    Args:
        scope_or_name: `string` or `VariableScope`, the scope to open.
        inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
        kernel: `4-D Tensor`, [kernel_height, kernel_width, in_channels, out_channels] kernel.
        bias: `1-D Tensor`, [out_channels] bias.
        strides: list of `ints`, length 4, the stride of the sliding window for each dimension of `inputs`.
        activation: activation function to be used (default: `tf.nn.relu`).
        use_bn: `bool`, whether or not to include batch normalization in the layer.
        is_training: `bool`, whether or not the layer is in training mode. This is only used if `use_bn` == True.
        use_lrn: `bool`, whether or not to include local response normalization in the layer.
        keep_prob: `double`, dropout keep prob.
        dropout_maps: `bool`, If true whole maps are dropped or not, otherwise single elements.
        padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.

    Returns:
        `4-D Tensor`, has the same type `inputs`.
    """
    with tf.variable_scope(scope_or_name):
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + kernel_shape[3]))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])),5e-2)
        kernel = tf.get_variable("weights", kernel_shape,
                                  initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(inputs, kernel, strides, padding='SAME', name='conv')
        bias = tf.get_variable("bias", kernel_shape[3],
                                 initializer=tf.constant_initializer(value=biasInit))
        outputs = tf.nn.bias_add(conv, bias, name='preActivation')
        if use_bn:
            # outputs = tf.layers.batch_normalization(outputs, axis=3, training=is_training, name="batchNorm")
            outputs = batch_norm(outputs, is_training=is_training, scale=True, fused=True, scope="batchNorm")
        if use_mvn:
            outputs = feat_norm(outputs, kernel_shape[3])
        if activation:
            outputs = activation(outputs, name='activation')
        if use_lrn:
            outputs = tf.nn.local_response_normalization(outputs, name='localResponseNorm')
        if dropout_maps:
            conv_shape = tf.shape(outputs)
            n_shape = tf.stack([conv_shape[0], 1, 1, conv_shape[3]])
            outputs = tf.nn.dropout(outputs, keep_prob, noise_shape=n_shape)
        else:
            outputs = tf.nn.dropout(outputs, keep_prob)
        return outputs


def dil_conv2d_bn_lrn_drop(scope_or_name,
                           inputs,
                           kernel_shape,
                           rate,
                           activation=tf.nn.relu,
                           use_bn=False,
                           use_mvn=False,
                           is_training=True,
                           use_lrn=True,
                           keep_prob=1.0,
                           dropout_maps=False,
                           initOpt=0):
    """Adds a 2-D convolutional layer given 4-D `inputs` and `kernel` with optional BatchNorm, LocalResponseNorm and Dropout.

    Args:
        scope_or_name: `string` or `VariableScope`, the scope to open.
        inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
        kernel: `4-D Tensor`, [kernel_height, kernel_width, in_channels, out_channels] kernel.
        bias: `1-D Tensor`, [out_channels] bias.
        rate: `int`, Dilation factor.
        activation: activation function to be used (default: `tf.nn.relu`).
        use_bn: `bool`, whether or not to include batch normalization in the layer.
        is_training: `bool`, whether or not the layer is in training mode. This is only used if `use_bn` == True.
        use_lrn: `bool`, whether or not to include local response normalization in the layer.
        keep_prob: `double`, dropout keep prob.
        dropout_maps: `bool`, If true whole maps are dropped or not, otherwise single elements.
        padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.

    Returns:
        `4-D Tensor`, has the same type `inputs`.
    """
    with tf.variable_scope(scope_or_name):
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + kernel_shape[3]))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])),5e-2)
        kernel = tf.get_variable("weights", kernel_shape,
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.atrous_conv2d(inputs, kernel, rate=rate, padding='SAME')
        bias = tf.get_variable("bias", kernel_shape[3],
                               initializer=tf.constant_initializer(value=0.1))
        outputs = tf.nn.bias_add(conv, bias, name='preActivation')
        if use_bn:
            # outputs = tf.layers.batch_normalization(outputs, axis=3, training=is_training, name="batchNorm")
            outputs = batch_norm(outputs, is_training=is_training, scale=True, fused=True, scope="batchNorm")
        if use_mvn:
            outputs = feat_norm(outputs, kernel_shape[3])
        if activation:
            outputs = activation(outputs, name='activation')
        if use_lrn:
            outputs = tf.nn.local_response_normalization(outputs, name='localResponseNorm')
        if dropout_maps:
            conv_shape = tf.shape(outputs)
            n_shape = tf.stack([conv_shape[0], 1, 1, conv_shape[3]])
            outputs = tf.nn.dropout(outputs, keep_prob, noise_shape=n_shape)
        else:
            outputs = tf.nn.dropout(outputs, keep_prob)
        return outputs



def deconv2d_bn_lrn_drop(scope_or_name, inputs, kernel_shape, out_shape, subS=2, activation=tf.nn.relu,
                       use_bn=False,
                       use_mvn=False,
                       is_training=True,
                       use_lrn=False,
                       keep_prob=1.0,
                       dropout_maps=False,
                       initOpt=0):
    with tf.variable_scope(scope_or_name):
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + kernel_shape[3]))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])),5e-2)
        kernel = tf.get_variable("weights", kernel_shape,
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", kernel_shape[2],
                               initializer=tf.constant_initializer(value=0.1))
        conv=tf.nn.conv2d_transpose(inputs, kernel, out_shape, strides=[1, subS, subS, 1], padding='SAME', name='conv')
        outputs = tf.nn.bias_add(conv, bias, name='preActivation')
        if use_bn:
            # outputs = tf.layers.batch_normalization(outputs, axis=3, training=is_training, name="batchNorm")
            outputs = batch_norm(outputs, is_training=is_training, scale=True, fused=True, scope="batchNorm")
        if use_mvn:
            outputs = feat_norm(outputs, kernel_shape[3])
        if activation:
            outputs = activation(outputs, name='activation')
        if use_lrn:
            outputs = tf.nn.local_response_normalization(outputs, name='localResponseNorm')
        if dropout_maps:
            conv_shape = tf.shape(outputs)
            n_shape = tf.stack([conv_shape[0], 1, 1, conv_shape[3]])
            outputs = tf.nn.dropout(outputs, keep_prob, noise_shape=n_shape)
        else:
            outputs = tf.nn.dropout(outputs, keep_prob)
        return outputs


def downsample_avg(images, sub):
    return tf.nn.avg_pool(images, ksize=[1, sub, sub, 1], strides=[1, sub, sub, 1], padding='SAME', name='down_avg')


def upsample_simple(images, shape_out, up, numClasses):
    filter_up = tf.constant(1.0, shape=[up, up, numClasses, numClasses])
    return tf.nn.conv2d_transpose(images, filter_up,
                                  output_shape=shape_out,
                                  strides=[1, up, up, 1])


def feat_norm(input, dimZ):
    beta = tf.get_variable('beta', shape=(dimZ,), initializer=tf.constant_initializer(value=0.0))
    gamma = tf.get_variable('gamma', shape=(dimZ,), initializer=tf.constant_initializer(value=1.0))
    output,_, _ = tf.nn.fused_batch_norm(input, gamma, beta)
    return output


def separable_rnn(images, num_filters_out, scope=None, keep_prob=1.0, cellType='LSTM'):
  """Run bidirectional LSTMs first horizontally then vertically.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output layer depth
    nhidden: hidden layer depth
    scope: optional scope name

  Returns:
    (num_images, height, width, num_filters_out) tensor
  """
  with tf.variable_scope(scope, "SeparableLstm", [images]):
    with tf.variable_scope("horizontal"):
      if 'LSTM' in cellType:
        cell_fw = LSTMCell(num_filters_out, use_peepholes=True, state_is_tuple=True)
        cell_bw = LSTMCell(num_filters_out, use_peepholes=True, state_is_tuple=True)
      if 'GRU' in cellType:
        cell_fw = GRUCell(num_filters_out)
        cell_bw = GRUCell(num_filters_out)
      hidden = horizontal_cell(images, num_filters_out, cell_fw, cell_bw, keep_prob=keep_prob, scope=scope)
    with tf.variable_scope("vertical"):
      transposed = tf.transpose(hidden, [0, 2, 1, 3])
      if 'LSTM' in cellType:
        cell_fw = LSTMCell(num_filters_out, use_peepholes=True, state_is_tuple=True)
        cell_bw = LSTMCell(num_filters_out, use_peepholes=True, state_is_tuple=True)
      if 'GRU' in cellType:
        cell_fw = GRUCell(num_filters_out)
        cell_bw = GRUCell(num_filters_out)
      output_transposed = horizontal_cell(transposed, num_filters_out, cell_fw, cell_bw, keep_prob=keep_prob, scope=scope)
    output = tf.transpose(output_transposed, [0, 2, 1, 3])
    return output


def horizontal_cell(images, num_filters_out, cell_fw, cell_bw, keep_prob=1.0, scope=None):
  """Run an LSTM bidirectionally over all the rows of each image.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output depth
    scope: optional scope name

  Returns:
    (num_images, height, width, num_filters_out) tensor, where
  """
  with tf.variable_scope(scope, "HorizontalGru", [images]):
    sequence = images_to_sequence(images)

    shapeT = tf.shape(sequence)
    sequence_length = shapeT[0]
    batch_sizeRNN = shapeT[1]
    sequence_lengths = tf.to_int64(
      tf.fill([batch_sizeRNN], sequence_length))
    forward_drop1 = DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
    backward_drop1 = DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
    rnn_out1, _ = tf.nn.bidirectional_dynamic_rnn(forward_drop1, backward_drop1, sequence, dtype=tf.float32,
                                                  sequence_length=sequence_lengths, time_major=True,
                                                  swap_memory=True, scope=scope)
    rnn_out1 = tf.concat(rnn_out1, 2)
    rnn_out1 = tf.reshape(rnn_out1, shape=[-1, batch_sizeRNN, 2, num_filters_out])
    output_sequence = tf.reduce_sum(rnn_out1, axis=2)
    batch_size=tf.shape(images)[0]
    output = sequence_to_images(output_sequence, batch_size)
    return output

def images_to_sequence(tensor):
  """Convert a batch of images into a batch of sequences.

  Args:
    tensor: a (num_images, height, width, depth) tensor

  Returns:
    (width, num_images*height, depth) sequence tensor
  """
  transposed = tf.transpose(tensor, [2, 0, 1, 3])

  shapeT = tf.shape(transposed)
  shapeL = transposed.get_shape().as_list()
  # Calculate the ouput size of the upsampled tensor
  n_shape = tf.stack([
      shapeT[0],
      shapeT[1]*shapeT[2],
      shapeL[3]
  ])
  reshaped = tf.reshape(transposed, n_shape)
  return reshaped

def sequence_to_images(tensor, num_batches):
  """Convert a batch of sequences into a batch of images.

  Args:
    tensor: (num_steps, num_batchesRNN, depth) sequence tensor
    num_batches: the number of image batches

  Returns:
    (num_batches, height, width, depth) tensor
  """

  shapeT = tf.shape(tensor)
  shapeL = tensor.get_shape().as_list()
  # Calculate the ouput size of the upsampled tensor
  height = tf.to_int32(shapeT[1] / num_batches)
  n_shape = tf.stack([
      shapeT[0],
      num_batches,
      height,
      shapeL[2]
  ])

  reshaped = tf.reshape(tensor, n_shape)
  return tf.transpose(reshaped, [1, 2, 0, 3])
