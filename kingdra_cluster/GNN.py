from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy.sparse as sp
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from keras.layers import Input
from keras.models import Model
from pygsp import graphs
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score
from spektral.layers import MinCutPool, DiffPool
from spektral.layers.convolutional import GCSConv
from spektral.utils.convolution import normalized_adjacency
from tqdm import tqdm

def GNN(A,X):
    # 使用graph_hi构造图邻接矩阵A
    np.random.seed(0)  # for reproducibility
    ITER = 10000
    # Parameters
    P = OrderedDict([
        ('es_patience', ITER),
        ('dataset', ['cora']),  # 'cora', 'citeseer', 'pubmed', 'cloud', or 'synth'
        ('H_', [None]),
        ('n_channels', [16]),
        ('learning_rate', [5e-4])
    ])
    ############################################################################
    # LOAD DATASET
    ############################################################################

    A = np.maximum(A, A.T)
    A = sp.csr_matrix(A, dtype=np.float32)

    X = X.todense()
    n_feat = X.shape[-1]

    ############################################################################
    # GNN MODEL
    ############################################################################
    X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_feat), name='X_in'))
    A_in = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), name='A_in', sparse=True)
    # S_in = Input(tensor=tf.placeholder(tf.int32, shape=(None,), name='segment_ids_in'))

    A_norm = normalized_adjacency(A)
    X_1 = GCSConv(P['n_channels'],
                  kernel_initializer='he_normal',
                  activation='elu')([X_in, A_in])

    pool1, adj1, C = MinCutPool(k=n_classes, h=P['H_'], activation='elu', return_mask=True)([X_1, A_in])

    model = Model([X_in, A_in], [pool1, adj1, C])
    model.compile('adam', None)

    ############################################################################
    # TRAINING
    ############################################################################
    # Setup
    sess = K.get_session()
    loss = model.total_loss
    opt = tf.train.AdamOptimizer(learning_rate=P['learning_rate'])
    train_step = opt.minimize(loss)

    # Initialize all variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Fit layer
    tr_feed_dict = {X_in: X,
                    A_in: sp_matrix_to_sp_tensor_value(A_norm)}

    best_loss = np.inf
    patience = P['es_patience']
    tol = 1e-5
    for _ in tqdm(range(ITER)):
        outs = sess.run([train_step, model.losses[0], model.losses[1], C], feed_dict=tr_feed_dict)
        # c = np.argmax(outs[3], axis=-1)
        if outs[1] + outs[2] + tol < best_loss:
            best_loss = outs[1] + outs[2]
            patience = P['es_patience']
        else:
            patience -= 1
            if patience == 0:
                break

    ############################################################################
    # RESULTS
    ############################################################################
    C_ = sess.run([C], feed_dict=tr_feed_dict)[0]
    c = np.argmax(C_, axis=-1)
    K.clear_session()
    return c