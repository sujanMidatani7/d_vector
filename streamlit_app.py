import streamlit as st
import torch


from sklearn.metrics.pairwise import cosine_similarity
import librosa
import numpy as np

# # Load the encoder classifier model
# # def compute_similarity(x1, x2):
# #     x1_tensor = torch.tensor(x1)
# #     x2_tensor = torch.tensor(x2)
# #     dot_product = torch.sum(x1_tensor * x2_tensor, dim=-1)
# #     norm_x1 = torch.norm(x1_tensor, dim=-1)
# #     norm_x2 = torch.norm(x2_tensor, dim=-1)
# #     cosine_similarity = dot_product / (norm_x1 * norm_x2)z
# #     return cosine_similarity
# def extract_mfcc_features(audio_file):
#     # Load the audio file
#     y, sr = librosa.load(audio_file,sr=16000)
# #     y=y.numpy()
#     # Compute the MFCC features
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=360)
#     mfcc_delta = librosa.feature.delta(mfcc)
#     mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
#     mfcc_features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
#     st.write(mfcc_features.shape)
#     n_features, n_samples = mfcc_features.shape
#     mfcc_features = mfcc_features.reshape(n_features, -1)
#     mfcc_features = mfcc_features.reshape(n_features*n_samples, 2)


#     return mfcc_features

# def load_audio(file):
#     # Load audio file using Torchaudio
    
#         waveform, sample_rate = 
#         return waveform, sample_rate
    

def compute_dvector(audio_file):
    # Apply PCA to reduce the dimensionality of the MFCC features
#     mfcc_features=mfcc_features.reshape(-1,1)
#     st.write(mfcc_features.shape)
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
    return d
def compute_similarity(dvector1, dvector2):
    # Compute the cosine similarity between the d-vectors
    similarity = cosine_similarity([dvector1], [dvector2])

    return similarity


# Define Streamlit app
st.title("Audio Analysis")
st.write("Comparision of two audio samples using xvectors.")

# Create audio input component
audio_file1 = st.file_uploader("Choose 1st audio  file", type=["mp3", "wav", "flac"])

dvector1=torch.rand(2,1)
dvector2=torch.rand(2,1)
if audio_file1 is not None:
    
    dvector1 = compute_dvector(audio_file1)

audio_file2=st.file_uploader("Choose 2nd audio  file", type=["mp3", "wav", "flac"])
if audio_file2 is not None:
#     mfcc_features2 = extract_mfcc_features(audio_file2)
    dvector2 = compute_dvector(audio_file2)

st.write("the similarity of the given two audio files is:")
similarity = compute_similarity(dvector1, dvector2)
st.write(similarity)
try:
    if similarity>=0.9:
        st.write("both are equal")
    else :
        st.write("both are not equal")
except:
    pass
