import os
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sequence_generator import SequenceGenerator
from keras.utils import multi_gpu_model

def build_model(vocab_size, seq_length=30, batch_size=32):
    model = Sequential()

    model.add(Embedding(vocab_size, 200, input_length=seq_length, trainable=True))
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

class GenericLM():
    def __init__(self, vocab_size, mapping, seq_length=30, batch_size=32,
                ckpt_path='./ckpt', model_path='./model', mode_name='left2right'):
        self.vocab_size = vocab_size
        self.mapping = mapping
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.ckpt_path = ckpt_path
        self.model_path = model_path
        self.mode_name = mode_name

        if os.path.exists(os.path.join(self.model_path, 'GenericLM_%s.model'%self.mode_name)):
            self.model = load_model(os.path.join(self.model_path, 'GenericLM_%s.model'%self.mode_name))
        else:
            self.model = build_model(self.vocab_size, self.seq_length, self.batch_size)
            self.load_ckpt()

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

    def load_ckpt(self):
        ckpt_file = os.listdir(self.ckpt_path)
        ckpt_file = list(filter(lambda x: x[-2:] == 'h5', ckpt_file))
        if ckpt_file:
            self.model.load_weights(os.path.join(self.ckpt_path, ckpt_file[-1]))

    def predict(self, return_prob_table=False):
        pass

class GenericLM_multi():
    def __init__(self, vocab_size, mapping, seq_length=30, batch_size=32,
                ckpt_path='./ckpt', model_path='./model', mode_name='left2right'):
        self.vocab_size = vocab_size
        self.mapping = mapping
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.ckpt_path = ckpt_path
        self.model_path = model_path
        self.mode_name = mode_name

        if os.path.exists(os.path.join(self.model_path, 'GenericLM_%s.model'%self.mode_name)):
            self.model = load_model(os.path.join(self.model_path, 'GenericLM_%s.model'%self.mode_name))
        else:
            self.model = build_model(self.vocab_size, self.seq_length, self.batch_size)
            self.load_ckpt()
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

    def load_ckpt(self):
        ckpt_file = os.listdir(self.ckpt_path)
        ckpt_file = list(filter(lambda x: x[-2:] == 'h5', ckpt_file))
        if ckpt_file:
            self.model.load_weights(os.path.join(self.ckpt_path, ckpt_file[-1]))

    def predict(self, return_prob_table=False):
        pass
