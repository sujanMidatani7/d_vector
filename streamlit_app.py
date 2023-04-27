import streamlit as st
import torch


from sklearn.metrics.pairwise import cosine_similarity
import librosa
import numpy as np

def compute_dvector(audio_file):

    audio, sr = librosa.load(audio_file, sr=16000)
    frame_length = int(sr * 0.025)  # 25 ms
    hop_length = int(sr * 0.010)  # 10 ms
    n_fft = 512
    n_mels = 40
    n_components = 10

    

    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,hop_length=hop_length, n_mels=n_mels)

    # Compute the d-vector by averaging the normalized PCA-transformed MFCC features
    S = np.log(S + 1e-9)

# compute delta and delta-delta
    delta = librosa.feature.delta(S, width=3)
    delta_delta = librosa.feature.delta(S, order=2, width=3)

# concatenate features
    features = np.concatenate((S, delta, delta_delta), axis=0)
    mean = np.mean(features, axis=1)
    cov = np.cov(features)
    
    U, s, Vh = np.linalg.svd(cov)
    A = np.dot(np.dot(U, np.diag(np.sqrt(s))), U.T)
    x = np.mean(features, axis=1)
# d = np.dot(np.dot(A, U.T), (x - mean))
    d=np.dot(A, U.T)
    d = np.dot(np.dot(A, U.T), (x))
    return d
def compute_similarity(dvector1, dvector2):
    # Compute the cosine similarity between the d-vectors
    similarity = cosine_similarity([dvector1], [dvector2])[0][0]
    return similarity


# Define Streamlit app
st.title("Audio Analysis")
st.write("Comparision of two audio samples using xvectors.")

# Create audio input component
audio_file1 = st.file_uploader("Choose 1st audio  file", type=["mp3", "wav", "flac"])

dvector1=torch.rand(120,1)
dvector2=torch.rand(120,1)
if audio_file1 is not None:
    
    dvector1 = compute_dvector(audio_file1)

audio_file2=st.file_uploader("Choose 2nd audio  file", type=["mp3", "wav", "flac"])
if audio_file2 is not None:
#     mfcc_features2 = extract_mfcc_features(audio_file2)
    dvector2 = compute_dvector(audio_file2)

st.write("the similarity of the given two audio files is:")
if audio_file2 is not None and audio_file1 is not None:
    similarity = compute_similarity(dvector1, dvector2)
    st.write(similarity)
    try:
        if similarity>=0.9:
            st.write("both are equal")
        else :
            st.write("both are not equal")
    except:
        pass
