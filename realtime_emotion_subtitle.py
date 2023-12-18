import sounddevice as sd
import numpy as np
import librosa
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from pyAudioAnalysis import ShortTermFeatures
from utils.prosodic_extraction_utils import *
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Routine for capturing an speech audio from microphone and classifying the speaker`s emotion')
parser.add_argument('--modelPath', type=str, default="./results/aug_mfcc/best_model_fold2.keras", help='Path for the model [NEEDS TO BE RELATIVE TO THE PROJECT FOLDER, USING ./]')
parser.add_argument('--duration', type=int, default=5, help='Audio capture duration in integer seconds')
parser.add_argument('--wavPath', type=str, default='', help='Path for a wavfile if the user wants to test it instead of a captured audio from the microphone')
args = parser.parse_args()
print("=> Iniciando com parâmetros:")
print(f'--modelPath: {args.modelPath}')
print(f'--duration: {args.duration}')
print(f'--wavPath: {args.wavPath}')
    
full_extract_method = args.modelPath.split('/')[2]
extract_method = full_extract_method.replace('_', '').replace('aug', '').replace('1', '')
encoded_labels = {0: 'felicidade', 1: 'neutro', 2: 'raiva', 3: 'tristeza'}

# Not used, but necessary for the model to understand the imported keras file,
# because custom f1 metric was incorporated as the evaluation metric for it
def f1_metric(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=1)
    y_true_labels = tf.cast(y_true, dtype=tf.int64)
    return tf.py_function(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'), 
                        (y_true_labels, y_pred_labels), 
                        tf.double)
    
def test_microphone(audio_data, samplerate):
    import soundfile as sf
    sf.write('tone.wav', audio_data, samplerate)
   
def test_wavfile(path):
    return librosa.load(path, sr=None)

# Extract features using one of 5 methods [mfcc, melspec, rms, pyaudioanalysis, praat-parselmouth]
def extract_features(y, sr):
    if extract_method == 'mfcc':
        n_mfcc = 13
        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # You can adjust the number of coefficients (n_mfcc)
        return format_features(features)
    elif extract_method == 'melspec':
        features = librosa.feature.melspectrogram(y=y, sr=sr)
        return format_features(features)
    elif extract_method == 'rms':
        features = librosa.feature.rms(y=y)
        return format_features(features)
    elif extract_method == 'pyaudioanalysis':
        win, step = 0.050 * sr, 0.025 * sr
        features, feat_names = ShortTermFeatures.feature_extraction(y, sr, win, step)
        features_mean = np.mean(features, axis=1)
        df = pd.DataFrame()
        for index, attribute in enumerate(feat_names):
            df.at[0, attribute] = features_mean[index]
        return df
    elif extract_method == 'praat':
        y_float32 = y.astype(np.float32)
        sound = parselmouth.Sound(y_float32, sampling_frequency=sr)
        df = pd.DataFrame()
        attributes = {}

        intensity_attributes = get_intensity_attributes(sound)[0]
        pitch_attributes = get_pitch_attributes(sound)[0]
        attributes.update(intensity_attributes)
        attributes.update(pitch_attributes)

        hnr_attributes = get_harmonics_to_noise_ratio_attributes(sound)[0]
        gne_attributes = get_glottal_to_noise_ratio_attributes(sound)[0]
        attributes.update(hnr_attributes)
        attributes.update(gne_attributes)

        df['local_jitter'] = None
        df['local_shimmer'] = None
        df.at[0, 'local_jitter'] = get_local_jitter(sound)
        df.at[0, 'local_shimmer'] = get_local_shimmer(sound)

        spectrum_attributes = get_spectrum_attributes(sound)[0]
        attributes.update(spectrum_attributes)

        formant_attributes = get_formant_attributes(sound)[0]
        attributes.update(formant_attributes)

        for attribute in attributes:
            df.at[0, attribute] = attributes[attribute]
            
        rearranged_columns = df.columns.tolist()[-1:] + df.columns.tolist()[:-1]
        df = df[rearranged_columns]
        return df
    
# Arranges features in a dataframe understandable by the model
def format_features(features):
    mean_features = np.mean(features, axis=1)
    std_features = np.std(features, axis=1)
    max_features = np.max(features, axis=1)
    min_features = np.min(features, axis=1)
    
    features_stat = np.concatenate([mean_features, std_features, max_features, min_features])
    
    columns = [f'Mean_{i}' for i in range(features.shape[0])]
    columns += [f'Std_{i}' for i in range(features.shape[0])]
    columns += [f'Min_{i}' for i in range(features.shape[0])]
    columns += [f'Max_{i}' for i in range(features.shape[0])]
    
    df = pd.DataFrame(columns=columns)
    for i, feat in enumerate(features_stat):
        df.at[0, columns[i]] = feat

    return df

# Performs feature extraction, normalization and the predicts class
def classify(y, sr):
    df = extract_features(y, sr).drop(['label', 'sound_filepath'], axis=1, errors='ignore')
    
    # Normalization is done around the dataset features mean, therefore it is necessary to access them
    dataset = pd.read_csv(f'./features/emoUERJ_features/{full_extract_method}_features.csv').drop(['label', 'sound_filepath'], axis=1, errors='ignore')
    all_features = pd.concat([df, dataset], ignore_index=True)
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    # Separate only the normalized feature vector of the sample
    sample_features = all_features[0, :].reshape(1, -1)
    sample_features = sample_features.reshape((sample_features.shape[0], sample_features.shape[1], 1))
    
    model = keras.models.load_model(args.modelPath, custom_objects={'f1_metric': f1_metric})
    predictions = model.predict(sample_features).tolist()[0]
    
    predicted_percentages = []
    for pred in predictions:
        predicted_percentages.append(pred)
        
    return predicted_percentages

# Records audio from the microphone and sends for classification
def record_and_classify():

    if args.wavPath != '':
        audio_data_float, sample_rate = test_wavfile(args.wavPath)
    else:
        print("Nenhum arquivo encontrado...")
        exit()

    predicted_percentages = classify(audio_data_float, sample_rate)
    print(f"As porcentagens de emoção são:")
    for i, pred in enumerate(predicted_percentages): 
        print(f"---{encoded_labels[i]}: {pred}")

while True:

    
    user_input = input("Pressione Enter para gravar (ou 'q' para sair): ")
    
    if user_input.lower() == 'q':
        print("Finalizando o programa.")
        break
    
    if not user_input:  # Check if Enter key is pressed
        record_and_classify()
