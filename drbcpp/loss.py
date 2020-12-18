import tensorflow as tf


def pairwise_ranking_crossentropy_loss(y_true, y_pred):
    """
    :@param y_true: [batch = target_betweenness | src_ids | tgt_ids]
    :@param y_pred: [batch = pred_betweenness]
    The original DrBC implementation uses 5*N src_id,tgt_id pairs from a graph of N nodes
    """
    pred_betweenness = y_pred
    target_betweenness = tf.slice(y_true, begin=(0, 0), size=(-1, 1))
    src_ids = tf.cast(tf.reshape(tf.slice(y_true, begin=(0, 1), size=(-1, 5)), (-1,)), 'int32')
    tgt_ids = tf.cast(tf.reshape(tf.slice(y_true, begin=(0, 6), size=(-1, 5)), (-1,)), 'int32')

    labels = tf.nn.embedding_lookup(target_betweenness, src_ids) - tf.nn.embedding_lookup(target_betweenness, tgt_ids)
    preds = tf.nn.embedding_lookup(pred_betweenness, src_ids) - tf.nn.embedding_lookup(pred_betweenness, tgt_ids)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=tf.sigmoid(labels))
