import librosa
import numpy as np
a, b = 0.765, 1
def cal_OTAC(tempo_in, tempo_out):
    coefficients = [-2, -1, 0, 1, 2]
    tempo_options = [(2**coefficient) * tempo_in for coefficient in coefficients]
    tempo_distance = np.absolute([tempo_option - tempo_out for tempo_option in tempo_options])
    best_tempo = tempo_options[np.argmin(tempo_distance)]
    tempo_low = min(best_tempo, tempo_out)
    tempo_high = max(best_tempo, tempo_out)
    tempo_cross = ( (a - b)*tempo_low + np.sqrt( ((a-b)**2)*(tempo_low**2) + 4*a*b*tempo_high*tempo_low ) ) / (2 * a)
    print(tempo_cross)
    OTAC = {"in": tempo_cross/best_tempo, "out": tempo_cross/tempo_out}
    return OTAC

def cross_interval(y_in,y_out):
    _, beats_in = librosa.beat.beat_track(y_in)
    _, beats_out = librosa.beat.beat_track(y_out)
    cross_range = min(len(beats_in),len(beats_out))
    
    return

y, sr = librosa.load("moments.wav")
cross_interval(y)