import librosa
def calculate_tempo(y, sr, hop_length):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    return tempo
def get_beats(y, sr, hop_length):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    return beats