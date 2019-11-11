
import librosa
import librosa.display
import numpy as np

def separte_audio(y, sr):
    mag, phase = librosa.magphase(librosa.stft(y))
    nn_mag = librosa.decompose.nn_filter(mag,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    nn_mag = np.minimum(nn_mag, mag)
    mask_backgroud = librosa.util.softmask(nn_mag, 2 * (mag - nn_mag), 2)
    mask_vocal = librosa.util.softmask(mag - nn_mag, 10 * nn_mag, 2)
    backgroud = librosa.istft(mask_backgroud * mag * phase)
    vocal = librosa.istft(mask_vocal * mag * phase)
    return vocal, backgroud
def decompose_audio(y):
    harmonic, percussive = librosa.decompose.hpss(librosa.stft(y), margin=2)
    harmonic = librosa.istft(harmonic)
    percussive = librosa.istft(percussive)
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
