"""
    That module regroups the Machine Learning part of the project, the model
    definition, the training and prediction procedures, hyperparameter
    optimization and model export
"""
import sys
import os
import pickle
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (log_loss, roc_auc_score, average_precision_score,
        accuracy_score, f1_score, confusion_matrix, classification_report)
from tensorflow.python.framework import graph_util
from mlplatform.data_utils import (Vocab, load_dataset, batch_iterator,
        process_dataset, get_words_dataset)
plt.style.use('ggplot')

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PKG_DIR, 'artifacts')
STATIC_DIR = os.path.join(PKG_DIR, 'static')


class Config(object):
    '''
        The Config class is used to store various hyperparameters and
        classification task information.
        Model objects are passed a Config object at
        instantiation.
    '''

    def __init__(self, params):
        self.params = params
        self.batch_size = params['batch_size']
        self.word_embed_size = params['word_embed_size']
        self.text_embed_size = params['text_embed_size']
        self.hidden_sizes = params['hidden_sizes'] # hidden sizes of final MLP
        self.n_epochs = params['n_epochs']
        self.kp = params['kp'] # keep_prob for dropout
        self.lr = params['lr'] # learning rate
        self.l2 = params['l2'] # L2 regularization coeff
        self.label_size = params['label_size']

        # text length (texts longer than this are truncated, shorter are padded)
        self.text_len = params['text_len']


class LSTMNModel():
    '''
        The config class is used to store various hyperparameters and
        classification task information.
        Model objects are passed a Config() object at
        instantiation.
    '''

    def __init__(self,
                 config,
                 embedding_matrix_ini=None):

        self.config = config
        self.embedding_matrix_ini = embedding_matrix_ini

        self.model = None
        self.build_model()

    def build_model(self):
        '''
            Builds the computational graph of the Neural Network
        '''
        config = self.config
        k = config.text_embed_size
        L = config.text_len

        graph = tf.Graph()
        with graph.as_default():

            # input tensors
            self.text_ph = tf.placeholder(tf.int32, shape=[None, L],
                                           name='text')
            self.len_ph = tf.placeholder(tf.int32, shape=[None], name='len')
            self.labels_ph = tf.placeholder(tf.int32,
                                            shape=[None],
                                            name='label')

            # set keep_prob (for dropout) as a palceholder (so that at prediction
            # we can set it to 1.0
            self.kp_ph = tf.placeholder(tf.float32, name='kp')
            kp = self.kp_ph

            # set embedding matrix to pretrained embedding
            init_embeds = tf.constant(self.embedding_matrix_ini, dtype='float32')
            word_embeddings = tf.get_variable(
                    dtype='float32',
                    name='word_embeddings',
                    initializer=init_embeds,
                    trainable=False) # no fine-tuning of word embeddings

            x = tf.nn.embedding_lookup(word_embeddings, self.text_ph)
            x = tf.nn.dropout(x, kp)

            def lstmn(x, length, scope):
                '''
                    Implements the Long-Short Term Memory Network architecture
                '''
                with tf.variable_scope(scope):
                    # define all variables in the LSTMN model
                    W_h = tf.get_variable(name='W_h', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    W_hs = tf.get_variable(name='W_hs', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    W_x = tf.get_variable(name='W_x', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    b_M = tf.get_variable(name='b_M', initializer=tf.zeros([L, k]))
                    w = tf.get_variable(name='w', shape=[k, 1],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    b_a = tf.get_variable(name='b_a', initializer=tf.zeros([L]))

                    W_rnn_h_i = tf.get_variable(name='W_rnn_h_i', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    W_rnn_x_i = tf.get_variable(name='W_rnn_x_i', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    b_rnn_i = tf.get_variable(name='b_rnn_i', initializer=tf.zeros([k]))

                    W_rnn_h_f = tf.get_variable(name='W_rnn_h_f', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    W_rnn_x_f = tf.get_variable(name='W_rnn_x_f', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    b_rnn_f = tf.get_variable(name='b_rnn_f', initializer=tf.zeros([k]))

                    W_rnn_h_o = tf.get_variable(name='W_rnn_h_o', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    W_rnn_x_o = tf.get_variable(name='W_rnn_x_o', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    b_rnn_o = tf.get_variable(name='b_rnn_o', initializer=tf.zeros([k]))

                    W_rnn_h_c = tf.get_variable(name='W_rnn_h_c', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    W_rnn_x_c = tf.get_variable(name='W_rnn_x_c', shape=[k, k],
                            regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                    b_rnn_c = tf.get_variable(name='b_rnn_c', initializer=tf.zeros([k]))

                    c0 = tf.zeros([tf.shape(length)[0], k])
                    h0 = tf.zeros([tf.shape(length)[0], k])
                    hst_1 = tf.zeros([tf.shape(length)[0], k])
                    Cl, Hl = [c0], [h0]
                    for t in range(L):
                        Ct_1 = tf.stack(Cl, axis=1)
                        Ht_1 = tf.stack(Hl, axis=1)
                        H_mod = tf.reshape(Ht_1, [-1, k])

                        xt = x[:,t,:]
                        Xt = tf.reshape(tf.tile(xt, [1, t+1]), [-1, t+1, k])
                        Xt_mod = tf.reshape(Xt, [-1, k])

                        Hst_1 = tf.reshape(tf.tile(hst_1, [1, t+1]), [-1, t+1, k])
                        Hst_1_mod = tf.reshape(Hst_1, [-1, k])

                        Mt = tf.nn.tanh( tf.reshape(tf.matmul(H_mod, W_h),
                                             [-1, t+1, k]) +
                                         tf.reshape(tf.matmul(Xt_mod, W_x),
                                             [-1, t+1, k]) +
                                         tf.reshape(tf.matmul(Hst_1_mod, W_hs),
                                             [-1, t+1, k])  + b_M[:t+1])
                        Mt_w = tf.matmul(tf.reshape(Mt, [-1, k]), w)
                        alphat = tf.nn.softmax(tf.reshape(Mt_w, [-1, 1, t+1]) + b_a[:t+1])
                        cst = tf.reshape(tf.matmul(alphat, Ct_1), [-1, k])
                        hst = tf.reshape(tf.matmul(alphat, Ht_1), [-1, k])
                        hst_1 = hst

                        it = tf.sigmoid(tf.matmul(hst, W_rnn_h_i) +
                                        tf.matmul(xt, W_rnn_x_i) +
                                        b_rnn_i)
                        ft = tf.sigmoid(tf.matmul(hst, W_rnn_h_f) +
                                        tf.matmul(xt, W_rnn_x_f) +
                                        b_rnn_f)
                        ot = tf.sigmoid(tf.matmul(hst, W_rnn_h_o) +
                                        tf.matmul(xt, W_rnn_x_o) +
                                        b_rnn_o)
                        cht = tf.nn.tanh(tf.matmul(hst, W_rnn_h_c) +
                                         tf.matmul(xt, W_rnn_x_c) +
                                         b_rnn_c)

                        ct = ft*cst + it*cht
                        ht = ot*tf.nn.tanh(ct)

                        Cl.append(ct)
                        Hl.append(ht)
                return ( tf.transpose(tf.stack(Hl), [1, 0, 2]),
                         tf.transpose(tf.stack(Cl), [1, 0, 2]) )

            H, _ = lstmn(x, self.len_ph, 'lstmn')

            def get_last_relevant_output(out, seq_len):
                rng = tf.range(0, tf.shape(seq_len)[0])
                indx = tf.stack([rng, seq_len - 1], 1)
                last = tf.gather_nd(out, indx)
                return last

            h = get_last_relevant_output(H, self.len_ph)

            y = h

            # MLP classifier on top
            hidden_sizes = config.hidden_sizes
            for layer, size in enumerate(hidden_sizes):
                if layer > 0:
                    previous_size = hidden_sizes[layer-1]
                else:
                    previous_size = k
                W = tf.get_variable(name='W{}'.format(layer),
                        shape=[previous_size, size],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(config.l2))
                b = tf.get_variable(name='b{}'.format(layer),
                        initializer=tf.zeros([size]))
                y = tf.nn.relu(tf.matmul(y, W) + b)

            W_softmax = tf.get_variable(name='W_softmax',
                    shape=[hidden_sizes[-1], config.label_size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(config.l2))
            b_softmax = tf.get_variable(name='b_softmax',
                    initializer=tf.zeros([config.label_size]))

            logits = tf.matmul(y, W_softmax) + b_softmax

            if config.label_size == 1:
                logits = tf.squeeze(logits)
                labels = tf.cast(self.labels_ph, tf.float32)
                cross_entropy_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                        )
            else:
                cross_entropy_loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_ph, logits=logits)
                        )

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = cross_entropy_loss + tf.add_n(reg_losses)

            self.train_op = ( tf.train.AdamOptimizer(learning_rate=config.lr)
                                      .minimize(self.loss) )

            if config.label_size == 1:
                self.probs = tf.nn.sigmoid(logits, name='probs')
            else:
                self.probs = tf.nn.softmax(logits, name='probs')

            self.saver = tf.train.Saver()

        self.model = graph

    def create_feed_dict(self, text_batch, len_batch, label_batch, keep_prob):
        feed_dict = {
            self.text_ph: text_batch,
            self.len_ph: len_batch,
            self.labels_ph: label_batch,
            self.kp_ph: keep_prob
        }
        return feed_dict

    def run_epoch_training(self, session, X, lengths, y):
        '''
            Trains for one epoch
        '''
        verbose = True
        config = self.config
        kp = config.kp

        total_loss = []
        total_processed_examples = 0
        total_steps = int(np.ceil(X.shape[0] / self.config.batch_size))

        batch_it = batch_iterator(X, lengths, y, batch_size=config.batch_size)

        for step, (texts_b, len_b, y_b) in enumerate(batch_it):
            feed = self.create_feed_dict(texts_b, len_b, y_b, kp)
            loss, _ = session.run([self.loss, self.train_op], feed_dict=feed)

            total_processed_examples += len(y)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()
        return np.mean(total_loss), total_loss

    def predict(self, X, lengths):
        '''
            Computes predictions
        '''
        config = self.config
        # Deactivate dropout for prediction
        kp = 1.0
        results = []

        batch_it = batch_iterator(X, lengths, batch_size=config.batch_size, shuffle=False)

        with tf.Session(graph=self.model) as sess:
            self.saver.restore(sess, os.path.join(ARTIFACTS_DIR, 'lstmn.ckpt'))

            for step, (texts_b, len_b, y_b) in enumerate(batch_it):
                feed = self.create_feed_dict(texts_b, len_b, np.zeros_like(len_b), kp)
                preds = sess.run(self.probs, feed_dict=feed)
                results.extend(preds)
        return np.array(results)

    def fit(self, X, lengths, y):
        '''
            Trains the model, saves it and exports a plot of the training loss
        '''
        config = self.config
        complete_loss_history = []

        with tf.Session(graph=self.model) as sess:

            tf.global_variables_initializer().run()

            for epoch in range(config.n_epochs):
                print('Epoch {}'.format(epoch))
                train_loss, loss_history = self.run_epoch_training(sess, X, lengths, y)
                complete_loss_history.extend(loss_history)
                print('Training loss: {}'.format(train_loss))
                self.saver.save(sess, os.path.join(ARTIFACTS_DIR,
                                                   'lstmn.ckpt'))

        fig, ax = plt.subplots(1,1, figsize=(10,6))
        ax.plot(complete_loss_history)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Batch')
        ax.set_title('Training Loss')
        fig.savefig(os.path.join(STATIC_DIR, 'complete_loss_history.png'))
        plt.close(fig)


    def freeze_model(self):
        '''
            Freezes the model, writes it to disk for it to be deployed
        '''
        output_node_names = "probs"
        saver = tf.train.import_meta_graph(os.path.join(ARTIFACTS_DIR, 'lstmn.ckpt.meta'),
                                           clear_devices=True)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        sess = tf.Session()
        saver.restore(sess, os.path.join(ARTIFACTS_DIR, 'lstmn.ckpt'))
        output_graph_def = graph_util.convert_variables_to_constants(
                    sess,
                    input_graph_def,
                    output_node_names.split(",")
        )
        output_graph = os.path.join(ARTIFACTS_DIR, 'lstmn.pb')
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        sess.close()


def generate_params(n_configs, params_dict):
    '''
        Generates a list of Config objects to perform random
        hyperparameter search

        Args:
            n_configs (int): number of configs to generate
            params_dict (dict): dictionary with parameter names (string) as
            keys and distributions (scipy.stats.distribution) or lists (lists)
            or single value of parameters to sample from. If a list is
            provided as a value, the candidate parameter is uniformaly sampled

        Returns:
            list(Config)
    '''
    list_configs = []
    for _ in range(n_configs):
        parameters = {}
        for k, v in params_dict.items():
            try:
                val = v.rvs()
            except AttributeError:
                try:
                    val = v[np.random.randint(0, len(v))]
                except TypeError:
                    val = v
            parameters[k] = val

        config = Config(parameters)
        list_configs.append(config)
    return list_configs


def parameter_search(vocab, train_data, valid_data,
        n_configs, params_dict):
    '''
        Implements the parameter search to find the best set of parameters
        based on the score on the validation data

        Args:
            vocab (Vocab): number of configs to generate
            train_data (tuple of texts, lengths, labels): training data to fit the models
            valid_data (tuple of texts, lengths, labels): validation data to
            evaluate the models
            n_configs (int): number of parameter sets to try
            params_dict (dict): dictionary with parameter names (string) as
            keys and distributions (scipy.stats.distribution) or lists (lists)
            or single value of parameters to sample from. If a list is
            provided as a value, the candidate parameter is uniformaly sampled

        Returns:
            (best score, best Config, best model, dictionary of evaluation
            metrics values)
            tuple(float, Config, LSTMNModel, dict)
    '''
    text_train, len_train, y_train = train_data
    text_valid, len_valid, y_valid = valid_data

    list_params = generate_params(n_configs, params_dict)

    best_score = None
    for i, config in enumerate(list_params):
        print('------------- Test model {}'.format(i))
        print('Model parameters:\n{}'.format(config.params))
        model = LSTMNModel(config, embedding_matrix_ini=vocab.embedding_matrix)
        model.fit(text_train, len_train, y_train)
        preds = model.predict(text_valid, len_valid)

        if params_dict['label_size'] == 1:
            score = roc_auc_score(y_valid, preds)
        else:
            score = f1_score(y_valid, np.argmax(preds, 1), average='weighted')
        print('score:\t{}'.format(score))

        if (best_score is None) or (score > best_score):
            best_score = score
            best_config = config
            best_model = model
            best_preds = preds
    print('----')
    print('Best model parameters:\n{}'.format(best_config.params))

    # computing metrics
    metrics = {}
    if params_dict['label_size'] == 1:
        metrics['log_loss'] = log_loss(y_valid, best_preds)
        metrics['roc_auc'] = roc_auc_score(y_valid, best_preds)
        metrics['avg_precision'] = average_precision_score(y_valid, best_preds)
    else:
        metrics['log_loss'] = log_loss(y_valid, best_preds)
        metrics['accuracy'] = accuracy_score(y_valid, np.argmax(best_preds, 1))
        metrics['f1'] = f1_score(y_valid, np.argmax(best_preds, 1), average='weighted')
    print(metrics)
    return best_score, best_config, best_model, metrics



def run_training():
    label_to_ind = process_dataset(0.2)
    vocab = Vocab()
    vocab.construct(get_words_dataset())
    vocab.build_embedding_matrix()

    label_size = 1 if len(label_to_ind) == 2 else len(label_to_ind)
    text_len = 128

    # Load the training set
    text_train, len_train, y_train = load_dataset(vocab, text_len, 'train')

    # Load the validation set
    text_valid, len_valid, y_valid = load_dataset(vocab, text_len, 'valid')

    params_dict = {
                     'batch_size': 16,
                     'word_embed_size': 50,
                     'text_embed_size': 50,
                     'hidden_sizes':[[128, 64, 32], [128, 32]],
                     'n_epochs': 1,
                     'kp': [0.95, 1.0],
                     'lr': 0.001,
                     'l2': [0.00002, 0.00005, 0.0001, 0.0002],
                     'label_size': label_size,
                     'text_len': text_len}

    best_score, best_config, best_model, metrics = parameter_search(vocab,
            (text_train, len_train, y_train),
            (text_valid, len_valid, y_valid),
            1, params_dict)
    best_model.freeze_model()

    # add ind_to_label to the config to display predictions properly
    best_config.ind_to_label = {v: k for k, v in label_to_ind.items()}

    with open(os.path.join(ARTIFACTS_DIR,'vocab.pkl'), 'wb') as fp:
        pickle.dump(vocab, fp)
    with open(os.path.join(ARTIFACTS_DIR,'config.pkl'), 'wb') as fp:
        pickle.dump(best_config, fp)

    return metrics
