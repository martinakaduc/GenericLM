import os
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sequence_generator import SequenceGenerator
from keras.utils import multi_gpu_model
from text_utils import *
import h5py
import numpy as np

def build_model(vocab_size, seq_length=30, batch_size=32):
    model = Sequential()

    model.add(Embedding(vocab_size, 200, input_length=seq_length, trainable=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    model.summary()

    return model

def load_multigpu_checkpoint_weights(model, h5py_file):
    """
    Loads the weights of a weight checkpoint from a multi-gpu
    keras model.

    Input:

        model - keras model to load weights into

        h5py_file - path to the h5py weights file

    Output:
        None
    """

    print("Setting weights...")
    with h5py.File(h5py_file, "r") as file:

        # Get model subset in file - other layers are empty
        weight_file = file["sequential_1"]

        for layer in model.layers:

            try:
                layer_weights = weight_file[layer.name]
                print('Loading %s layer...' % layer.name)
            except:
                # No weights saved for layer
                continue

            try:
                weights = []
                # Extract weights
                for term in layer_weights:
                    if isinstance(layer_weights[term], h5py.Dataset):
                        # Convert weights to numpy array and prepend to list
                        if layer.name == 'lstm_1':
                            if term == 'bias:0':
                                weights.insert(0, np.array(layer_weights[term]))
                            elif term =='kernel:0':
                                weights.insert(0, np.array(layer_weights[term]))
                            else:
                                weights.insert(1, np.array(layer_weights[term]))
                        else:
                            weights.insert(0, np.array(layer_weights[term]))

                # Load weights to model
                layer.set_weights(weights)

            except Exception as e:
                print(e)
                print("Error: Could not load weights for layer: " + layer.name)

class GenericLM():
    def __init__(self, vocab_size, mapping, seq_length=30, batch_size=32, multi_gpu=False,
                ckpt_path='./ckpt', model_path='./model', mode_name='left2right'):
        self.vocab_size = vocab_size
        self.mapping = mapping
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.ckpt_path = ckpt_path
        self.model_path = model_path
        self.mode_name = mode_name

        if os.path.exists(os.path.join(self.model_path, 'GenericLM_%s.model'%self.mode_name)):
            print("Loading saved model...")
            self.model = load_model(os.path.join(self.model_path, 'GenericLM_%s.model'%self.mode_name))
        else:
            self.model = build_model(self.vocab_size, self.seq_length, self.batch_size)
            self.load_ckpt()

        if multi_gpu:
            self.model = multi_gpu_model(self.model)

    def fit(self, corpus, epochs, ckpt_period=1):
        optimizer = Adam(lr=5e-4, decay=5e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        checkpoint = ModelCheckpoint(os.path.join(self.ckpt_path, 'GenericLM_{epoch:03d}.h5'), period=ckpt_period, save_weights_only=True)
        early_stop = EarlyStopping(monitor='loss', patience=12)

        sequenece_genrator = SequenceGenerator(corpus, self.seq_length, self.mapping, batch_size=self.batch_size)

        self.model.fit_generator(generator=sequenece_genrator,
                                epochs=epochs,
                                callbacks=[checkpoint, early_stop])

        self.model.save(os.path.join(self.model_path, 'GenericLM_%s.model'%self.mode_name))

    def get_model(self):
        return self.model

    def load_ckpt(self):
        ckpt_file = os.listdir(self.ckpt_path)
        ckpt_file = list(filter(lambda x: x[-2:] == 'h5', ckpt_file))
        if ckpt_file:
            print("Restoring model from checkpoint...")
            self.model.load_weights(os.path.join(self.ckpt_path, ckpt_file[-1]))
            # load_multigpu_checkpoint_weights(self.model, os.path.join(self.ckpt_path, ckpt_file[-1]))

    def predict(self, X, return_prob_table=False, return_label=True):
        if return_prob_table:
            return self.model.predict_proba(X)

        else:
            next_char_predict = self.model.predict_classes(X)
            if return_label:
                next_char_predict = decode_sequence(self.mapping, next_char_predict)
            return list(next_char_predict)
