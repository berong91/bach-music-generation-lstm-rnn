#!/usr/bin/env python
# coding: utf-8
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from keras.engine.input_layer import InputLayer
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten, Reshape
from keras.models import Sequential
from music21 import instrument, note, chord, stream

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
    model.summary()
    return model


def generate():
    """ Generate a piano midi file """
    # load the notes used to train the model
    with open('model/classic-bach-4-notes-harmonization/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
        logger.info(f'Loading notes: {len(notes[0])}')

    # Get all pitch names
    pitchnames = sorted(set([n for voice in notes for n in voice]))
    # Get all pitch names
    n_vocab = len(set([n for voice in notes for n in voice]))

    logger.info('Initiating music generation process.......')

    network_input = get_inputSequences(notes, pitchnames, n_vocab)
    normalized_input = network_input / float(n_vocab)
    model = create_network(normalized_input, n_vocab)
    logger.info('Loading Model weights.....')
    model.load_weights('model/classic-bach-4-notes-harmonization/weights.best.music3.hdf5')
    logger.info('Model Loaded')
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)


def get_inputSequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    for i in range(0, len(notes[0]) - sequence_length, 1):
        network_input.append([])
        for j in range(len(notes)):
            network_input[i].append([])

    # create input sequences and the corresponding outputs
    for i in range(len(notes)):
        for j in range(0, len(notes[i]) - sequence_length, 1):
            sequence_in = notes[i][j: j + sequence_length]
            network_input[j][i].append([note_to_int[char] for char in sequence_in])

    n_features = len(network_input[1])
    n_patterns = len(network_input)

    # reshape the input into a format comatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, n_features))

    return (network_input)


def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # Pick a random integer
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    # pick a random sequence from the input as a starting point for the prediction
    pattern = list(network_input[start])
    prediction_output = [[], [], [], []]

    logger.info('Generating notes........')

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 4))
        prediction_input = prediction_input / float(n_vocab)
        prediction_input = np.asarray(prediction_input).astype('float32')

        prediction = model.predict(prediction_input, verbose=0)

        list_index = []
        for i in range(len(prediction[0])):
            # Predicted output is the argmax(P(h|D))
            index = np.argmax(prediction[0][i])

            # Mapping the predicted interger back to the corresponding note
            result = int_to_note[index]

            # Storing the predicted output
            prediction_output[i].append(result)

            list_index.append(index)

        pattern.append(list_index)
        # Next input to the model
        pattern = pattern[1:len(pattern)]

    logger.info('Notes Generated...')
    return prediction_output


def create_midi(prediction_output) -> str:
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    filename = f'{str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))}.midi'
    combined_score = stream.Score(id=f'generated score {filename}')

    part_id = 0
    for voice in prediction_output:
        part_id += 1
        voice_part = stream.Part(id=f'Part {part_id}')
        offset = 0

        # create note and chord objects based on the values generated by the model
        for pattern in voice:
            if ('.' in pattern) or pattern.isdigit():  # pattern is a chord
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    if part_id == 1:
                        new_note.storedInstrument = instrument.Soprano()
                    elif part_id == 2:
                        new_note.storedInstrument = instrument.Alto()
                    elif part_id == 3:
                        new_note.storedInstrument = instrument.Tenor()
                    elif part_id == 4:
                        new_note.storedInstrument = instrument.Bass()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                voice_part.append(new_chord)
            else:  # pattern is a note
                new_note = note.Note(pattern)
                new_note.offset = offset
                if part_id == 1:
                    new_note.storedInstrument = instrument.Soprano()
                elif part_id == 2:
                    new_note.storedInstrument = instrument.Alto()
                elif part_id == 3:
                    new_note.storedInstrument = instrument.Tenor()
                elif part_id == 4:
                    new_note.storedInstrument = instrument.Bass()
                voice_part.append(new_note)

                offset += 0.5

        combined_score.insert(0, voice_part)

    logger.info('Saving Output file as midi....')
    path = combined_score.write('midi', f'output/classic-bach-4-notes-harmonization/{filename}')
    combined_score.show()
    return path


if __name__ == '__main__':
    #### Generate a new music
    path = generate()
    logger.info(f'Output file: {path}')
