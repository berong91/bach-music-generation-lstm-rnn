Bach style music generation
-------------
## Requirements
- Python 3.8 or above
- Installing all the packages under requirement.txt

## Preconfig music21
- To visually display a Midi file, install MuseScore.
- run this command to config music21 library to detech MuseScore:
    ```python
    from music21 import *
    configure.run()
    ```
## Baseline model
- Base model is named `classic-bach-1-note-generation`
- `python training.py` will start the training model process
- `python testing.py` will start the music generating process
- the model is saved to corresponded folder under `model/` and `output/`

## Project model
- Base model is named `classic-bach-4-notes-harmonization`
- `python training_4features.py` will start the training model process
- `python testing_4features.py` will start the music generating process
- the model is saved to corresponded folder under `model/` and `output/`

## Output MIDI File
- All the midi files are stored under `output/`
- The `testing` script will always create a new sample, and try to display the sample on MuseScore
