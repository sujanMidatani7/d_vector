# d_vector
```markdown
# Audio Analysis with D-Vectors

This is a Streamlit application that compares two audio samples using d-vectors. It utilizes the Pinecone similarity search service for finding the nearest speaker based on the computed d-vectors.

## Prerequisites

- Python 3.7 or higher

## Installation

1. Clone the repository:

```shell
git clone https://github.com/sujanMidatani7/d_vector.git
```

2. Navigate to the project directory:

```shell
cd d_vector
```

3. Install the required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

1. Make sure you have obtained a Pinecone API key. If not, sign up for a Pinecone account and create an API key.

2. Open the `dVectorSA.py` file and replace `'f9571b23-70be-4556-893a-7342b0bb51d1'` in the `pinecone.init()` function with your Pinecone API key.

3. Run the Streamlit app:

```shell
streamlit run dVectorSA.py
```

4. The application will open in your browser. You can select the first audio file using the file uploader component.

5. After selecting the audio file, the app will compute the d-vector for the audio file and find the nearest speaker using the Pinecone index.

6. The app will display the nearest speaker's ID and score.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [GNU License](LICENSE).



