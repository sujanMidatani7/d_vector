import streamlit as st
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Load the encoder classifier model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
def cosine_similarity(x1, x2):
    dot_product = torch.sum(x1 * x2, dim=-1)
    norm_x1 = torch.norm(x1, dim=-1)
    norm_x2 = torch.norm(x2, dim=-1)
    cosine_similarity = dot_product / (norm_x1 * norm_x2)
    return cosine_similarity

def load_audio(file):
    # Load audio file using Torchaudio
    
        waveform, sample_rate = torchaudio.load(file)
        return waveform, sample_rate
    

def analyze_audio(file):
    waveform, sample_rate = load_audio(file)
    if waveform is None:
        return

    # Encode the audio signal using the x-vector model
    embeddings_xvect = classifier.encode_batch(waveform)
    return embeddings_xvect
    # Display the embeddings
#     st.write("The x-vector embeddings are:")
#     st.write(embeddings_xvect)

# Define Streamlit app
st.title("Audio Analysis")
st.write("Comparision of two audio samples using xvectors.")

# Create audio input component
audio_file1 = st.file_uploader("Choose 1st audio  file", type=["mp3", "wav", "flac"])
xvect1=torch.rand(2, 3,1)
xvect2=torch.rand(2, 3,1)
# Analyze audio properties when file is uploaded
if audio_file1 is not None:
    xvect1=analyze_audio(audio_file1)
audio_file2=st.file_uploader("Choose 2nd audio  file", type=["mp3", "wav", "flac"])
if audio_file2 is not None:
    xvect2=analyze_audio(audio_file2)
st.write("the similarity of the given two audio files is:")
st.write(cosine_similarity(xvect1,xvect2)[0][0])
