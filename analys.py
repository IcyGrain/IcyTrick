import utils
import scale
import tempo

def get_audios(path):
    audios = utils.read_file(path)
    return audios
def get_series(audio, sr):
    y = utils.load_audio(path=audio, sr=sr)
    return y
def get_scales(y, sr, plot_tag):
    harmonic, percussive = scale.decompose_audio(y)

    # harmonic = utils.lowpass_filter(harmonic,6,200,sr)
    if plot_tag:
        utils.scale_plot(harmonic, sr, "harmonic")

    chromagram = scale.chroma_constantQ(harmonic, sr)

    if plot_tag:
        utils.chroma_plot(chromagram, sr, "constantQ")

    # chromagram = utils.median_filter(y,100)
    # utils.scale_plot(chromagram, sr, "median_filter")

    # table_s = utils.scales_matrix()
    # table_t = utils.triads_matrix()
    table_q = utils.quints_matrix()

    chord = scale.multi_chord(chromagram, table_q)
    if plot_tag:
        utils.chroma_plot(chord, sr, "chord")
    keys = scale.maxbit_column(chord)
    if plot_tag:
        utils.chroma_plot(keys, sr, "keys")
    results = scale.sum_maxbit(keys)
    scales = utils.scale_labels(results)
    return scales
def get_tempo(y, sr, hop_length):
    bpm = tempo.calculate_tempo(y, sr, hop_length)
    return bpm
def analyse(info_tag=True, plot_tag=False, path=None, sr=22050, hop_length =512, json_interval=10, json_delete=True):
    audios = get_audios(path)
    result_list = utils.load_json()
    for i, audio in enumerate(audios):
        file_path, file_name = utils.path_split(audio)
        if i < len(result_list):
            print("Already Analyzed :", file_name)
        else:
            if info_tag:
                print("Now Is Analyzing :", file_name)
            y = get_series(audio, sr)
            if info_tag:
                print("\tTempo Detecting...")
            bpm = get_tempo(y, sr, hop_length)
            if info_tag:
                print("\tScales Detecting...")
            scales = get_scales(y, sr, plot_tag)
            result_list.append({'audio': file_name, 'path': file_path, 'tempo': bpm, 'scales': scales})
            if i % json_interval == 0:
                utils.save_json(result_list)
                print("\nSaveing data\n")
        if info_tag:
            print("Completed:  {}%   Remaining:   {}/{}".format(round((i+1)/len(audios) * 100, 4), len(audios)-i-1, len(audios)))
    if json_delete:
        utils.delete_json()
    return result_list
