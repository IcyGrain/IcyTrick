
import librosa
import librosa.display
import numpy as np

def decompose_audio(y):
    harmonic, percussive = librosa.effects.hpss(y)
    return harmonic, percussive

def chroma_constantQ(y, sr):
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
    return chromagram
def multi_chord(chromagram, table):
    chord = librosa.util.normalize(np.dot(table, chromagram),axis=1)
    return chord
def maxbit_column(chord):
    max_index = chord.argmax(0);
    keys = np.zeros(chord.shape)
    for i, column in enumerate(keys.T):
        column[max_index[i]] = 1
    return keys
def sum_maxbit(keys):
    results = [sum(row) for row in keys]
    return results
