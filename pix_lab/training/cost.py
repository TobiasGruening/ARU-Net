import tensorflow as tf
import numpy as np

def get_cost(logits, tgt, kwargs={}):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """

    n_class = tf.shape(logits)[3]
    cost_name = kwargs.get("cost_name", "cross_entropy")
    act_name = kwargs.get("act_name", "softmax")


    if act_name is "softmax":
        act = tf.nn.softmax
    if act_name is "sigmoid":
        act = tf.nn.sigmoid
    if act_name is "identity":
        act = tf.identity

    if not cost_name is "cross_entropy":
        prediction = act(logits)

    print("Cost Type: " + cost_name)

    if cost_name is "cross_entropy":
        class_weights = kwargs.get("class_weights", None)
        if class_weights is not None:
            print("Class Weights: " + str(class_weights))
            class_weights_tf = tf.constant(np.array(class_weights, dtype=np.float32))
            flat_logits = tf.reshape(logits, [-1, n_class])
            flat_labels = tf.reshape(tgt, [-1, n_class])
            weight_map = tf.multiply(flat_labels, class_weights_tf)
            weight_map = tf.reduce_sum(weight_map, axis=1)

            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                               labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)

            loss = tf.reduce_mean(weighted_loss)

        else:
            flat_logits = tf.reshape(logits, [-1, n_class])
            flat_labels = tf.reshape(tgt, [-1, n_class])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                         labels=flat_labels))
    elif cost_name is "dice":
        igore_last_channel = kwargs.get("igore_last_channel", True)
        eps = 1e-5
        if igore_last_channel:
            prediction = tf.slice(prediction, [0, 0, 0, 0], [-1, -1, -1, n_class - 1])
            tgt = tf.slice(tgt, [0, 0, 0, 0], [-1, -1, -1, n_class - 1])
        intersection = tf.reduce_sum(prediction * tgt)
        union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(tgt)
        loss = -(2 * intersection / (union))
    elif cost_name is "dice_mean":
        igore_last_channel = kwargs.get("igore_last_channel", True)
        eps = 1e-5
        uPred = tf.unstack(prediction,axis=3)
        uTgt = tf.unstack(tgt,axis=3)
        if igore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        dices=[]
        for aCH in range(0,len(uPred)):
            intersection = tf.reduce_sum(uPred[aCH] * uTgt[aCH])
            union = eps + tf.reduce_sum(uPred[aCH]) + tf.reduce_sum(uTgt[aCH])
            aDice = -(2 * intersection / (union))
            dices.append(aDice)
        loss = tf.reduce_mean(dices)
    elif cost_name is "mse":
        igore_last_channel = kwargs.get("igore_last_channel", True)
        uPred = tf.unstack(prediction, axis=3)
        uTgt = tf.unstack(tgt, axis=3)
        if igore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        mses = []
        for aCH in range(0, len(uPred)):
            aRMSE = tf.losses.mean_squared_error(uTgt[aCH], uPred[aCH])
            mses.append(aRMSE)
        loss = 0.0

        class_weights = kwargs.get("class_weights", None)
        if class_weights is not None:
            print("Class Weights: " + str(class_weights))
            norm = 0
            for aCH in range(0,len(mses)):
                loss += class_weights[aCH]*mses[aCH]
                norm += class_weights[aCH]
        else:
            for aCH in range(0,len(mses)):
                loss += mses[aCH]

    elif cost_name is "mse_mean":
        igore_last_channel = kwargs.get("igore_last_channel", True)
        uPred = tf.unstack(prediction, axis=3)
        uTgt = tf.unstack(tgt, axis=3)
        if igore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        mses = []
        sums=[]
        globSum = 0.0
        for aCH in range(0, len(uPred)):
            aRMSE = tf.losses.mean_squared_error(uTgt[aCH],uPred[aCH])
            mses.append(aRMSE)
            aSum = tf.reduce_sum(uTgt[aCH])
            globSum += aSum
            sums.append(aSum)
        loss = tf.cond(tf.equal(globSum, 0.0),
                              lambda: tf.reduce_mean(mses),
                              lambda: get_weighted_mean(mses, sums, globSum))
    elif cost_name is "nse":
        igore_last_channel = kwargs.get("igore_last_channel", True)
        eps = 1e-5
        norm = eps + tf.reduce_sum(tgt)+tf.reduce_sum(prediction)
        if igore_last_channel:
            prediction = tf.slice(prediction, [0, 0, 0, 0], [-1, -1, -1, n_class - 1])
            tgt = tf.slice(tgt, [0, 0, 0, 0], [-1, -1, -1, n_class - 1])
            norm = eps + tf.reduce_sum(tgt) + tf.reduce_sum(prediction)
        loss = tf.reduce_sum(tf.squared_difference(tgt, prediction))
        loss = loss/norm
    elif cost_name is "nse_mean":
        igore_last_channel = kwargs.get("igore_last_channel", True)
        eps = 1e-5
        uPred = tf.unstack(prediction, axis=3)
        uTgt = tf.unstack(tgt, axis=3)
        if igore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        mses = []
        for aCH in range(0, len(uPred)):
            norm = eps + tf.reduce_sum(uTgt[aCH]) + tf.reduce_sum(uPred[aCH])
            aRMSE = tf.reduce_sum(tf.squared_difference(uTgt[aCH], uPred[aCH]))/norm
            mses.append(aRMSE)
        loss = tf.reduce_mean(mses)
    elif cost_name is "combined":
        igore_last_channel = kwargs.get("igore_last_channel", True)

        flat_logits = tf.reshape(logits, [-1, n_class])
        flat_labels = tf.reshape(tgt, [-1, n_class])
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                      labels=flat_labels))
        eps = 1e-5
        uPred = tf.unstack(prediction, axis=3)
        uTgt = tf.unstack(tgt, axis=3)
        if igore_last_channel:
            uPred = uPred[:-1]
            uTgt = uTgt[:-1]
        dices = []
        for aCH in range(0, len(uPred)):
            intersection = tf.reduce_sum(uPred[aCH] * uTgt[aCH])
            union = eps + tf.reduce_sum(uPred[aCH]) + tf.reduce_sum(uTgt[aCH])
            aDice = -(2 * intersection / (union))
            dices.append(aDice)
        loss2 = tf.reduce_mean(dices)

        loss = loss1 + loss2
    elif cost_name is "cross_entropy_sum":
        flat_logits = tf.reshape(logits, [-1, n_class])
        flat_labels = tf.reshape(tgt, [-1, n_class])
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                      labels=flat_labels))
    else:
        raise ValueError("Unknown cost function: " % cost_name)

    regularizer = kwargs.get("regularizer", None)
    if regularizer is not None:
        costs = []
        for var in tf.trainable_variables():
            if 'weights' in var.op.name:
                costs.append(tf.nn.l2_loss(var))
        loss += (regularizer * tf.add_n(costs))

    return loss

def get_weighted_mean(mses, sums, globSum):
    loss = 0.0
    for aCH in range(0, len(sums)):
        aLoss = (1.0-sums[aCH]/globSum) * mses[aCH]
        loss += aLoss
    return loss