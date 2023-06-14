import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import numpy as np
import pinecone

# Initialize Pinecone with API key
pinecone.init(api_key="f9571b23-70be-4556-893a-7342b0bb51d1", environment="us-central1-gcp")
pc = pinecone

# Create an index in Pinecone
index = pc.Index('id-index')

def compute_dvector(audio_file):
    st.write(audio_file)

    # Extract speaker name from the audio file name
    speaker_name = (audio_file.name.split('/')[-1]).split('.')[0]

    # Load the audio file and set the sampling rate to 16000
    audio, sr = librosa.load(audio_file, sr=16000)

    # Defining parameters for feature extraction
    frame_length = int(sr * 0.025)  # 25 ms
    hop_length = int(sr * 0.010)  # 10 ms
    n_fft = 512
    n_mels = 40
    n_components = 10

    # Computing the mel spectrogram of the audio signal
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Apply logarithm to stabilize the spectrogram
    S = np.log(S + 1e-9)

    # Compute delta and delta-delta features
    delta = librosa.feature.delta(S, width=3)
    delta_delta = librosa.feature.delta(S, order=2, width=3)

    # Feature Contatenation
    features = np.concatenate((S, delta, delta_delta), axis=0)

    # The mean and covariance of the features
    mean = np.mean(features, axis=1)
    cov = np.cov(features)

    # Singular value decomposition (SVD) on the covariance matrix
    U, s, Vh = np.linalg.svd(cov)

    # Construct the affine transformation matrix A
    A = np.dot(np.dot(U, np.diag(np.sqrt(s))), U.T)

    # Compute the d-vector using the affine transformation and mean features
    x = np.mean(features, axis=1)
    d = np.dot(np.dot(A, U.T), (x))

    # Query the Pinecone index to find the nearest speaker
    dvector1 = index.query(
        vector=list(d),
        top_k=1,
        include_values=True
    )['matches'][0]

    # Return the nearest speaker's ID and score
    return "Nearest Speaker Found :: " + dvector1['id'] + " with score of :: " + str(dvector1['score'])

# Define Streamlit app
st.title("Audio Analysis")
st.write("Comparison of two audio samples using d-vectors.")

# Create audio input component for the first audio file
audio_file1 = st.file_uploader("Choose 1st audio file", type=["mp3", "wav", "flac", "m4a"])

if audio_file1 is not None:
    # Compute d-vector for the first audio file
    rs = compute_dvector(audio_file1)

    try:
        # Display the nearest speaker ID and score
        st.write("Nearest Speaker Found :: " + dvector1[1] + " with score of :: " + dvector1[0])
        st.write(rs)
    except:
        pass
