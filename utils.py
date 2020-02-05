import json

import librosa
import librosa.display
import scipy.ndimage
import os
import matplotlib.pyplot as plot
from operator import itemgetter
from scipy.signal import butter, lfilter


# IO utils
def read_file(path):
    if path is None:
        path = os.path.abspath(os.curdir)
        return [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.wav')]
    if os.path.isdir(path):
        return [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.wav')]
    elif os.path.isfile(path) and path.endswith('.wav'):
        return [path]
    else:
        return


def path_split(path):
    file_path, file_name = os.path.split(path)
    return file_path, file_name


def load_audio(path, sr):
    y, sr = librosa.load(path=path, sr=sr)
    return y


def write_audio(path, y, sr):
    librosa.output.write_wav(path, y, sr)


# JSON utils
def load_json():
    if not os.path.exists("data.json"):
        with open("data.json", "w") as f:
            json.dump([], f)
    with open("data.json", "r") as f:
        result_list = json.load(f)
        return result_list


def save_json(result_list):
    with open("data.json", "w") as f:
        json.dump(result_list, f)


def delete_json():
    os.remove("data.json")


# filter utils
def lowpass_filter(data, order, cutoff, sr):
    nyq = sr * 0.5
    normal_cutoff = cutoff / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='lowpass')
    y = lfilter(b, a, data)
    return y


def median_filter(data, size):
    y = []
    for point in data:
        y.append(scipy.ndimage.filters.median_filter(point, size, mode='constant'))
    return y


# plot utils
def scale_plot(y, sr, title):
    scale = librosa.feature.melspectrogram(y=y, sr=sr)
    log_scale = librosa.amplitude_to_db(scale)
    plot.figure(figsize=(12, 6))

    librosa.display.specshow(data=log_scale, sr=sr, y_axis='mel')
    plot.title('mel power spectrogram:' + title)
    plot.colorbar(format='%+02.0f dB')
    plot.tight_layout()
    plot.show()


def chroma_plot(y, sr, title):
    plot.figure(figsize=(12, 4))
    librosa.display.specshow(y, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plot.title('Chromagram:' + title)
    plot.colorbar()
    plot.tight_layout()
    plot.show()


# static matrix utils
def scales_matrix():
    mat = [[0.0] * 12 for i in range(12)]
    for i in range(12):
        mat[i][i] = 5
        mat[i][(i + 2) % 12] = 1
        mat[i][(i + 4) % 12] = 2
        mat[i][(i + 5) % 12] = 1
        mat[i][(i + 7) % 12] = 2
        mat[i][(i + 9) % 12] = 2
        mat[i][(i + 11) % 12] = 0

    return mat


def triads_matrix():
    mat = [[0.0] * 12 for i in range(24)]
    for i in range(12):
        mat[i][i] = 1
        mat[i][(i + 4) % 12] = 1
        mat[i][(i + 7) % 12] = 1

        mat[i + 12][i] = 1
        mat[i + 12][(i + 3) % 12] = 1
        mat[i + 12][(i + 7) % 12] = 1

    return mat


def quints_matrix():
    mat = [[0.0] * 12 for i in range(12)]
    for i in range(12):
        mat[i][i] = 3
        mat[i][(i + 7) % 12] = 1

    return mat


def scale_labels(results):
    labels = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    weights = [(4 * results[i % 12] + 2 * (results[(i + 5) % 12] + results[(i + 7) % 12]) + results[(i + 9) % 12]) for i
               in range(12)]
    scales = sorted(dict(zip(labels, weights / sum(weights))).items(), key=itemgetter(1), reverse=True)
    return scales
