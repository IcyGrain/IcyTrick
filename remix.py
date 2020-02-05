import librosa
import numpy as np
import pydub

import utils
import tempo

a, b = 0.765, 1


def cal_OTAC(tempo_in, tempo_out):
    coefficients = [-2, -1, 0, 1, 2]
    tempo_options = [(2 ** coefficient) * tempo_in for coefficient in coefficients]
    tempo_distance = np.absolute([tempo_option - tempo_out for tempo_option in tempo_options])
    best_tempo = tempo_options[np.argmin(tempo_distance)]
    tempo_low = min(best_tempo, tempo_out)
    tempo_high = max(best_tempo, tempo_out)
    tempo_cross = ((a - b) * tempo_low + np.sqrt(
        ((a - b) ** 2) * (tempo_low ** 2) + 4 * a * b * tempo_high * tempo_low)) / (2 * a)
    print(tempo_cross)
    OTAC = {"in": tempo_cross / best_tempo, "out": tempo_cross / tempo_out}
    return OTAC


def cross_interval(y_in, y_out, sr, hop_length, block):
    beats_in = tempo.get_beats(y_in, sr, hop_length)
    beats_out = tempo.get_beats(y_out, sr, hop_length)
    cross_range = min(len(beats_in), len(beats_out))
    # lowpass_in = utils.lowpass_filter(y_in, 20, 1500, sr)
    # lowpass_out = utils.lowpass_filter(y_in, 20, 1500, sr)
    scores = []
    for i in range(round(cross_range / 4)):
        score = 0

        for j in range(i):
            in_start = beats_in[-i - 1] * hop_length - block
            in_end = beats_in[-i - 1] * hop_length + block
            out_start = beats_out[i] * hop_length - block
            out_end = beats_out[i] * hop_length + block
            power_in = np.sum(np.abs(y_in[in_start:in_end]))
            power_out = np.sum(np.abs(y_out[out_start:out_end]))
            score += power_in * power_out
        scores.append(score / (i + 1))
    remix_point = np.argmax(scores)
    sheet_in = [beats_in[-remix_point], beats_in[-int(remix_point / 2)], beats_in[-1]]
    sheet_out = [beats_out[0], beats_out[int(remix_point / 2)], beats_out[remix_point]]
    print(sheet_in)
    print(sheet_out)


def main():
    sr = 44100
    hop_length = 512
    y_in = utils.load_audio("Awake.wav", sr)
    y_out = utils.load_audio("Light.wav", sr)
    cross_interval(y_in, y_out, sr, hop_length, 10)


if __name__ == '__main__':
    dub = pydub.AudioSegment.from_file("Awake.wav", format="wav", frame_rate=44100, channels=1)
    y, sr = librosa.load("Awake.wav", sr=44100)
    # sa = np.zeros((y.shape[1] * 2), dtype=y.dtype)
    # sa2dub = pydub.AudioSegment(data=sa.tobytes(), frame_rate=44100, sample_width=sa.dtype.itemsize,
    #                             channels=len(y.shape))
    # dub.export("dub.wav", format="wav")
    # sa2dub.export("sa2dub.wav", format="wav")
    samples = dub.get_array_of_samples()
    sample_float = librosa.util.buf_to_float(samples, n_bytes=2, dtype=np.float32)
    print(len(y))
    print(y)
    print(sample_float)
    arr = np.array([np.mean([sample_float[i * 2], sample_float[i * 2 + 1]]) for i in range(len(y))])
    dub.overlay()
    print(len(arr))

    print(arr)
