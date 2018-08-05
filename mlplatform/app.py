"""
    That module sets up the Flask app that handles the interface with the user
"""
import sys
import os
import pickle
from mlplatform.data_utils import Vocab, pad_text, encode_text
from mlplatform.trainer import Config, LSTMNModel, run_training
from flask import Flask, render_template, send_from_directory, request, redirect,url_for
from flask import jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PKG_DIR, 'artifacts')
DATASETS_DIR = os.path.join(PKG_DIR, 'datasets')
app.graph = None


def load_model():
    '''
        Loads the frozen tensorflow graph, config and vocab
    '''
    with tf.gfile.GFile(os.path.join(ARTIFACTS_DIR, 'lstmn.pb'), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="")

    app.graph = graph
    app.session = tf.Session(graph=app.graph)
    with open(os.path.join(ARTIFACTS_DIR, 'vocab.pkl'), 'rb') as f:
        app.vocab = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'config.pkl'), 'rb') as f:
        app.model_config = pickle.load(f)


@app.route('/')
def index():
    return 0


@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
        Prediction interface, allows the user to enter some text and get
        predicted probabilities using the trained model

        Returns:
            rendered template
    '''
    results = {}
    if request.method == 'POST':

        if app.graph is None:
            load_model()

        text = request.form['user_text']

        graph = app.graph
        session = app.session
        vocab = app.vocab
        model_config = app.model_config

        text_len = model_config.text_len

        encoded_text = encode_text(vocab, text)
        length = len(encoded_text)
        processed_text = np.array(
                pad_text(vocab, encoded_text, text_len)
                )[np.newaxis,:]

        y_pred = graph.get_tensor_by_name("probs:0")
        text_ph = graph.get_tensor_by_name("text:0")
        len_ph = graph.get_tensor_by_name("len:0")
        kp_ph = graph.get_tensor_by_name("kp:0")

        feed = {text_ph: processed_text,
                len_ph: np.array([length]),
                kp_ph: 1.0}
        preds = session.run(y_pred, feed_dict=feed)

        # put the results in the format: {label: predicted_probability}
        if model_config.label_size == 1:
            results[model_config.ind_to_label[1]] = preds
            results[model_config.ind_to_label[0]] = 1-preds
        else:
            for ind, label in model_config.ind_to_label.items():
                results[label] = preds[0][ind]

    return render_template('predict.html', results=results)


@app.route('/train',methods=['POST', 'GET'])
def training():
    '''
        Training interface, allows the user to upload a dataset, train a model
        and returns some validation set evaluation metrics and a plot of the
        training loss

        Returns:
            rendered template
    '''
    success = None
    loss_img = None
    metrics = None
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(DATASETS_DIR, 'dataset.tsv'))
        metrics = run_training()
        success = 1
        loss_img = '/static/complete_loss_history.png'

        # load the trained model
        load_model()
    return render_template('training.html',
            success=success,
            loss_img=loss_img,
            metrics=metrics)


if __name__ == '__main__':
    app.run()
