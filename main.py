import os
import streamlit as st
import pandas as pd
import numpy as np
import glob
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# Menambahkan CSS untuk mengubah warna
st.markdown(
    """
    <style>
    .icon {
        font-size: 50px;
        color: #ff4b4b;
    }
    .css-1d391kg {
        background-color: #333333;
        color: white;
    }
    body {
        background-color: #f0f0f0;
        color: #333333;
    }
    .custom-title {
        color: #ff6347;
        font-size: 40px;
    }
    .custom-text {
        color: #4682B4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_option('deprecation.showPyplotGlobalUse', False)

# TESTING PREPROCESS
def Trimming_padding(file_path, max_length=51200):
    audio, sample_rate = librosa.load(file_path, sr=None)
    audio, _ = librosa.effects.trim(audio)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        pad_width = max_length - len(audio)
        audio = np.pad(audio, pad_width=(0, pad_width), mode='constant', constant_values=0)
    return audio, sample_rate

def extract_mfcc(audio, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfccs.shape[1] > 100:
        mfccs = mfccs[:, :100]
    else:
        pad_width = 100 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    return mfccs

def extract_additional_features(audio, sample_rate, max_frames=100):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=12, n_fft=2048, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=2048, hop_length=512)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=512)

    def pad_feature(feature, max_frames):
        if feature.shape[1] > max_frames:
            return feature[:, :max_frames]
        else:
            pad_width = max_frames - feature.shape[1]
            return np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    chroma = pad_feature(chroma, max_frames)
    spectral_contrast = pad_feature(spectral_contrast, max_frames)
    zcr = pad_feature(zcr, max_frames)

    return chroma, spectral_contrast, zcr

def process_and_extract_features(uploaded_file, max_length=51200):
    audio, sample_rate = Trimming_padding(uploaded_file, max_length)
    mfccs = extract_mfcc(audio, sample_rate)
    chroma, spectral_contrast, zcr = extract_additional_features(audio, sample_rate)
    features = np.hstack([mfccs.flatten(), chroma.flatten(), spectral_contrast.flatten(), zcr.flatten()])

    # Load the scaler and transform features
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    features_scaled = scaler.transform([features])

    # Reshaping to the expected 3D format (samples, height, width, channels)
    height = 20  # Number of MFCC features
    width = features_scaled.shape[1] // height
    X = features_scaled.reshape(1, height, width, 1)  # Only one sample, hence the first dimension is 1
    return X

# Buat menu navigasi di sidebar
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih halaman:", ["Home", "Upload Audio", "Klasifikasi"])

if menu == "Home":
    st.markdown('<h1 class="custom-title">Selamat Datang di Sistem Pengenalan Emosi Ucapan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">Silakan pilih menu di sidebar untuk melanjutkan.</p>', unsafe_allow_html=True)
    st.markdown('<i class="icon fas fa-headphones"></i>', unsafe_allow_html=True)

    # Tampilkan gambar dataset
    st.title("Dataset")
    st.write("Pada pembangunan model ini digunakan 9.615 dataset dengan 5 label yaitu anger sebanyak 1.923 data, sad sebanyak 1.923 data, happy sebanyak 1.923 data, disgust sebanyak 1.923 data, dan fear sebanyak 1.923 data. Dataset pada penelitian ini didapat dari pengguna bernama Ahmed Abozaid.")
    st.write("Berikut adalah tautan ke dataset yang digunakan di Kaggle: https://www.kaggle.com/datasets/ahmedeabozaid/audio-sentiment-analysis/data.")
    st.title("Visualisasi Dataset Model")
    from PIL import Image
    image = Image.open(r'C:\Users\dayua\Documents\dataset\gambar\gambar2.jpg')
    st.image(image, caption='Deskripsi gambar', use_column_width=True)
    image_folder_path = os.path.expanduser("~/Documents/dataset")  # Ganti dengan path ke folder gambar kamu
    images = os.listdir(image_folder_path)
    for image_file in images:
        image_path = os.path.join(image_folder_path, image_file)
        if image_file.endswith(('png', 'jpg', 'jpeg', 'gif')):
            st.image(image_path, caption=image_file, use_column_width=True)
        
    dataset_folder_path = "/path/to/your/dataset/gambar"  # Ganti dengan path ke folder dataset di laptop Anda
  



elif menu == "Upload Audio":
    st.title("Upload Audio")
    st.markdown('<i class="icon fas fa-file-audio"></i>', unsafe_allow_html=True)

    # Unggah file audio
    uploaded_file = st.file_uploader("Upload file WAV", type=["wav"])

    if uploaded_file is not None:
        # Simpan file yang diunggah ke session state
        st.session_state['uploaded_file'] = uploaded_file

        # Putar file audio
        st.audio(uploaded_file, format='audio/wav')
   
    #visualisasi video
    def visualize_audio(file_path):
        # Load audio file
        y, sr = librosa.load(file_path)

        # Plot the audio signal
        st.title("Visualisasi Audio dengan Waveform")
        plt.figure(figsize=(14, 5))
        plt.title('Audio Signal')
        plt.plot(y)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        st.pyplot()

        # Plot the spectrogram
        st.title("Visualisasi Audio Spectogram")
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_db, x_axis='time', y_axis='log')
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        st.pyplot()

    # Jika file audio telah diunggah, tampilkan visualisasinya
    if 'uploaded_file' in st.session_state:
        file_path = st.session_state['uploaded_file']
        visualize_audio(file_path)

elif menu == "Klasifikasi":
    st.title("Klasifikasi Emosi Ucapan")
    st.markdown('<i class="icon fas fa-file-audio"></i>', unsafe_allow_html=True)

    if 'uploaded_file' in st.session_state:
        uploaded_file = st.session_state['uploaded_file']

         # Putar file audio
        st.audio(uploaded_file, format='audio/wav')

        # Button untuk memulai klasifikasi
        if st.button("Mulai Klasifikasi"):
            X = process_and_extract_features(uploaded_file)

            imported_model = keras.models.load_model("model.keras")
            # Predict the labels for your test data
            predictions = imported_model.predict(X)

            # Convert the predicted probabilities to class labels
            predicted_labels = np.argmax(predictions, axis=1)
            if predicted_labels == 0:
                emotion = "Anger"
            elif predicted_labels == 1:
                emotion = "Disgust"
            elif predicted_labels == 2:
                emotion = "Fear"
            elif predicted_labels == 3:
                emotion = "Happy"
            elif predicted_labels == 4:
                emotion = "Sad"
            st.write("Hasil Prediksi:", emotion)




# elif menu == "Klasifikasi":
#     st.title("Klasifikasi Emosi Ucapan")
#     st.markdown('<i class="icon fas fa-brain"></i>', unsafe_allow_html=True)

#     if 'uploaded_file' in st.session_state:
#         uploaded_file = st.session_state['uploaded_file']

#         # Putar file audio
#         st.audio(uploaded_file, format='audio/wav')

#         # Button untuk memulai klasifikasi
#         if st.button("Mulai Klasifikasi"):
#             # Tempatkan logika klasifikasi di sini
#             st.write("Klasifikasi sedang berjalan...")

#             def load(uploaded_files):
#                 # prepare empty list to store audio
#                 audio_librosa = []
#                 # load the audio data from uploaded files then append it to the list
#                 for uploaded_file in uploaded_files:
#                     audio, _ = librosa.load(uploaded_file, sr = 22550)
#                     audio_librosa.append(audio)

#                 # return the list
#                 return audio_librosa

#             # Gunakan fungsi load untuk memuat audio dari file yang diunggah
#             audio_data = load([uploaded_file])

#             # Ekstraksi fitur dari file audio yang diunggah
#             y, sr = audio_data[0], 22550
#             mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
#             mfccs = np.mean(mfccs.T, axis=0)  # Mengambil rata-rata MFCC
#             mfccs = mfccs.reshape(1, -1)  # Bentuk ulang menjadi satu baris

#             # Path ke file dataset
#             file_path = r'C:\Users\dayua\Documents\dataset\dataset.csv'  # Sesuaikan path dengan lokasi file Anda

#             if not os.path.exists(file_path):
#                 st.error(f"File tidak ditemukan: {file_path}")
#     else:
#         st.write("Silakan unggah file audio terlebih dahulu di menu ")
