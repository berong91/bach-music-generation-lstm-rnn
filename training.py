#!/usr/bin/env python
# coding: utf-8
import logging
import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from music21 import note, chord, corpus

#############################################
MODEL_NAME = 'music-generation'
#############################################
# LOGGING SETUP
#############################################
logger = logging.getLogger(MODEL_NAME)
logger.setLevel(logging.INFO)
# create handlers
fh = logging.FileHandler(f'{MODEL_NAME}_log.txt', mode='a', encoding=None, delay=False)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
#############################################
# PANDA CONFIG
#############################################
pd.options.mode.chained_assignment = None
############################################


songs = glob(fr'input/bach/*.mxl')

np.random.seed(42)
songs = np.random.choice(songs, size=10)


def get_notes():
    notes = []
    for file in songs:
        logger.info(os.path.basename(file))

        # midi = converter.parse(file) # converting .mid file to stream object
        # try:  # Given a single stream, partition into a part for each unique instrument
        #     parts = instrument.partitionByInstrument(midi)
        # except:
        #     pass

        midi = corpus.parse(f'bach/{os.path.splitext(os.path.basename(file))[0]}')
        parts = midi.parts

        notes_to_parse = [4]
        if parts:  # if parts has instrument
            for part in parts:
                if part.partName == 'Soprano':
                    notes_to_parse.extend(part.recurse())
                    break
            if not notes_to_parse:
                notes_to_parse.extend(parts[0].recurse())
        else:
            notes_to_parse.extend(midi.flat.notes)

        # if element is a note, extract pitch,
        # else if element is a chord, append the normal form
        # of the # chord (a list of integers) to the list of notes.
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif (isinstance(element, chord.Chord)):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('model/classic-bach-1-note-generation/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    sequence_length = 100

    # Extract the unique pitches in the list of notes.
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format comatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input
    network_input = network_input / float(n_vocab)

    # one hot encode the output vectors
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_in, n_vocab):
    """Create the model architecture"""
    model = Sequential()
    model.add(LSTM(128, input_shape=network_in.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def train(model, network_input, network_output, epochs):
    """
    Train the neural network
    """
    # Create checkpoint to save the best model weights.
    filepath = 'model/classic-bach-1-note-generation/weights.best.music3.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)

    model.fit(network_input, network_output, epochs=epochs, batch_size=32, callbacks=[checkpoint])


def train_network():
    """
    Get notes
    Generates input and output sequences
    Creates a model
    Trains the model for the given epochs
    """

    epochs = 200

    notes = get_notes()
    logger.info('Notes processed')

    n_vocab = len(set(notes))
    logger.info('Vocab generated')

    network_in, network_out = prepare_sequences(notes, n_vocab)
    logger.info('Input and Output processed')

    model = create_network(network_in, n_vocab)
    logger.info('Model created')
    # return model
    logger.info('Training in progress')
    train(model, network_in, network_out, epochs)
    logger.info('Training completed')


if __name__ == '__main__':
    ### Train the model
    train_network()
