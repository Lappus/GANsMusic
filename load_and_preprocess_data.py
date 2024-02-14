import os
import matplotlib.pyplot as plt
import numpy as np
#for loading and visualizing audio files
import librosa
import librosa.display
import tensorflow as tf

first_audio_path = "data/pianoTriadDataset/audio/"
first_audio_clips = os.listdir(first_audio_path)
tesT="data/preprocessed_data.npy"

second_audio_path = "data/pianoTriadDataset/audio_augmented_x10/"
second_audio_clips = os.listdir(second_audio_path)
#print("No. of .wav files in audio folder = ",len(audio_clips))

def load_and_preprocess_data(audio_path, audio_clips, audio_path2, audio_clips2, target_shape=(128, 128)):
    data = []

    for i in range(len(audio_clips)):
        audio_data, sr = librosa.load(audio_path + audio_clips[i], sr=16000) 
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    for i in range(len(audio_clips2)):
        audio_data, sr = librosa.load(audio_path2 + audio_clips2[i], sr=16000) 
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
        print('shape:', mel_spectrogram.shape)
        
        
    print('Data  shape:', np.shape(data))
    np.save("data/preprocessed_data.npy", data)
    print("Preprocessed data saved successfully.")

    return np.array(data)

if __name__ == "__main__":
    load_and_preprocess_data(audio_path=first_audio_path, audio_clips=first_audio_clips, audio_path2=second_audio_path, audio_clips2=second_audio_clips)
  
        



    