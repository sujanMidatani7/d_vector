import streamlit as st
import torch
import torchaudio
from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity
import librosa
import numpy as np

# Load the encoder classifier model

def compute_similarity(x1, x2):
    dot_product = torch.sum(x1 * x2, dim=-1)
    norm_x1 = torch.norm(x1, dim=-1)
    norm_x2 = torch.norm(x2, dim=-1)
    cosine_similarity = dot_product / (norm_x1 * norm_x2)
    return cosine_similarity
def extract_mfcc_features(audio_file):
    # Load the audio file
    y, sr = torchaudio.load(audio_file)

    # Compute the MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=360)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.concatenate((mfcc, mfcc_delta,mfcc_delta2), axis=0)

    return mfcc_features.T

# def load_audio(file):
#     # Load audio file using Torchaudio
    
#         waveform, sample_rate = 
#         return waveform, sample_rate
    

def compute_dvector(mfcc_features):
    # Apply PCA to reduce the dimensionality of the MFCC features
    pca = PCA(n_components=120)
    pca.fit(mfcc_features)
    mfcc_pca = pca.transform(mfcc_features)

    # Compute the mean and standard deviation of the PCA-transformed MFCC features
    mfcc_pca_mean = np.mean(mfcc_pca, axis=0)
    mfcc_pca_std = np.std(mfcc_pca, axis=0)

    # Normalize the PCA-transformed MFCC features
    mfcc_pca_norm = (mfcc_pca - mfcc_pca_mean) / mfcc_pca_std

    # Compute the d-vector by averaging the normalized PCA-transformed MFCC features
    dvector = np.mean(mfcc_pca_norm, axis=0)

    return dvector
# def compute_similarity(dvector1, dvector2):
#     # Compute the cosine similarity between the d-vectors
#     similarity = cosine_similarity([dvector1], [dvector2])[0][0]

    return similarity
def softmax(z):
    """Computes softmax function.
    z: array of input values.
    Returns an array of outputs with the same shape as z."""
    # For numerical stability: make the maximum of z's to be 0.
    shiftz = z - np.max(z)
    exps = np.exp(shiftz)
    return exps / np.sum(exps)

# Define Streamlit app
st.title("Audio Analysis")
st.write("Comparision of two audio samples using xvectors.")

# Create audio input component
audio_file1 = st.file_uploader("Choose 1st audio  file", type=["mp3", "wav", "flac"])
mfcc_features1=torch.rand(2, 3,1)
mfcc_features2=torch.rand(2, 3,1)
# Analyze audio properties when file is uploaded
dvector1=torch.rand(2, 3,1)
dvector2=torch.rand(2, 3,1)
if audio_file1 is not None:
    mfcc_features1 = extract_mfcc_features(audio_file1)
    dvector1 = compute_dvector(mfcc_features1)

audio_file2=st.file_uploader("Choose 2nd audio  file", type=["mp3", "wav", "flac"])
if audio_file2 is not None:
    mfcc_features2 = extract_mfcc_features(audio_file2)
    dvector2 = compute_dvector(mfcc_features2)

st.write("the similarity of the given two audio files is:")
similarity = compute_similarity(dvector1, dvector2)
st.write(similarity)
if similarity>=0:
    st.write("both are equal")
else :
    st.write("both are not equal")
