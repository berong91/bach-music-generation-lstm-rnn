#!/usr/bin/env python
# coding: utf-8
import logging
import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.engine.input_layer import InputLayer
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten, Reshape
from keras.models import Sequential
from music21 import note, chord, corpus
from tensorflow import one_hot

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


# np.random.seed(42)
# songs = np.random.choice(songs, size=10)


def is_valid_note(element):
    return isinstance(element, note.Note) or isinstance(element, chord.Chord)


def get_correct_notes(element) -> str:
    if isinstance(element, note.Note):
        return str(element.pitch)
    elif isinstance(element, chord.Chord):
        return '.'.join(str(n) for n in element.normalOrder)


def get_notes():
    notes = [[], [], [], []]
    count = 0
    for file in songs:
        logger.info(os.path.basename(file))

        # midi = converter.parse(file) # converting .mid file to stream object
        # try:  # Given a single stream, partition into a part for each unique instrument
        #     parts = instrument.partitionByInstrument(midi)
        # except:
        #     pass

        midi = corpus.parse(f'bach/{os.path.splitext(os.path.basename(file))[0]}')
        parts = midi.parts

        notes_to_parse = [[], [], [], []]
        if parts:  # if parts has instrument
            for part in parts:
                if part.partName == 'Soprano':
                    notes_to_parse[0].extend(part.recurse())
                elif part.partName == 'Alto':
                    notes_to_parse[1].extend(part.recurse())
                elif part.partName == 'Tenor':
                    notes_to_parse[2].extend(part.recurse())
                elif part.partName == 'Bass':
                    notes_to_parse[3].extend(part.recurse())
            if not notes_to_parse[0]:
                notes_to_parse[0].extend(parts[0].recurse())
            if not notes_to_parse[1]:
                notes_to_parse[0].extend(parts[0].recurse())
            if not notes_to_parse[2]:
                notes_to_parse[0].extend(parts[0].recurse())
            if not notes_to_parse[3]:
                notes_to_parse[0].extend(parts[0].recurse())
        else:
            notes_to_parse[0].extend(midi.flat.notes)
            notes_to_parse[1].extend(midi.flat.notes)
            notes_to_parse[2].extend(midi.flat.notes)
            notes_to_parse[3].extend(midi.flat.notes)

        # if element is a note, extract pitch,
        # else if element is a chord, append the normal form
        # of the # chord (a list of integers) to the list of notes.
        for _notes in zip(*notes_to_parse):
            soprano, alto, tenor, bass = _notes

            if is_valid_note(soprano):
                count += 1
            valid_note = is_valid_note(soprano) and is_valid_note(alto) and is_valid_note(tenor) and is_valid_note(bass)
            if valid_note:
                notes[0].append(get_correct_notes(soprano))
                notes[1].append(get_correct_notes(alto))
                notes[2].append(get_correct_notes(tenor))
                notes[3].append(get_correct_notes(bass))

    with open('model/classic-bach-4-notes-harmonization/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    logger.info(f'Valid Soprano notes: {count}')
    logger.info(f'Valid sequences: {len(notes[0])}')
    logger.info(f'Cut-off percentage: {(count - len(notes[0]))/count * 100:.2f}%')
    return notes


def prepare_sequences(notes, n_vocab, sequence_length=100):
    # Extract the unique pitches in the list of notes.
    pitchnames = sorted(set([n for voice in notes for n in voice]))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []
    for i in range(0, len(notes[0]) - sequence_length, 1):
        network_input.append([])
        network_output.append([])
        for j in range(len(notes)):
            network_input[i].append([])

    # create input sequences and the corresponding outputs
    for i in range(len(notes)):
        for j in range(0, len(notes[i]) - sequence_length, 1):
            sequence_in = notes[i][j: j + sequence_length]
            sequence_out = notes[i][j + sequence_length]

            network_input[j][i].append([note_to_int[char] for char in sequence_in])
            network_output[j].append(note_to_int[sequence_out])

    n_features = len(network_input[1])
    n_patterns = len(network_input)

    # reshape the input into a format comatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, n_features))

    # normalize input
    network_input = network_input / float(n_vocab)

    # one hot encode the output vectors
    total_token = set()
    for _ in network_output:
        total_token.update(_)
    # layer = CategoryEncoding(num_tokens=len(network_output) + 1, output_mode="multi_hot")

    network_output = one_hot(network_output, n_vocab)

    return (network_input, network_output)


def create_network(network_in, n_vocab):
    """Create the model architecture"""
    model = Sequential()
    model.add(InputLayer(input_shape=network_in.shape[1:]))
    model.add(Reshape(target_shape=(4, 100)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab * 4))
    model.add(Reshape((4, n_vocab)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary(print_fn=logger.info)
    return model


def train(model, network_input, network_output, epochs):
    """
    Train the neural network
    """
    # Create checkpoint to save the best model weights.
    filepath = 'model/classic-bach-4-notes-harmonization/weights.best.music3.hdf5'
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

    n_vocab = len(set([n for voice in notes for n in voice]))
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
