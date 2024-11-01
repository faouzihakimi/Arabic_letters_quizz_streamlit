
# Arabic Letters quizz with streamlit

## Overview

This repository contains a Streamlit app designed to help users learn and practice Arabic letter writing. The app uses Deep learning to recognize handwritten Arabic letters and provide feedback to users.

## Repository Structure

- `model_training.ipynb`: Jupyter notebook containing the model training process for Arabic character recognition. The obtained model is saved in the app/ folder.
- `app/`: Folder containing the Streamlit application files.
- `Dockerfile`: Instructions for building a Docker container for the app.
- `requirements.txt`: Python dependencies required for the project.
- `packages.txt`: System-level packages required for the project.
- `.gitignore`: To prevent uploading the python env and data.

## Features

- Interactive interface for practicing Arabic letter writing
- Real-time character recognition
- Score tracking

## Try the App Online

You can try the app directly by visiting our published Streamlit app at [Arabic letter quizz](https://arabiclettersquizzapp-lga9vypkfhbvyagvngbh69.streamlit.app/). No installation is required!

## Local installation

### Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/faouzihakimi/Arabic_letters_quizz_streamlit
   cd Arabic_letters_quizz_streamlit
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Docker Setup

1. Build the Docker image:
   ```
   docker build -t arabic_letter_quizz .
   ```

2. Run the Docker container:
   ```
   docker run -p 8501:8501 arabic_letter_quizz
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run streamlit_app/main.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Follow the on-screen instructions to start practicing Arabic letter writing.

## Model Training

The `model_training.ipynb` notebook contains the process for training the character recognition model. To retrain or modify the model:

1. Open the notebook in Jupyter or JupyterLab.
2. Follow the instructions within the notebook to prepare your dataset and train the model.
3. Export the trained model and update the relevant files in the `streamlit_app/` directory.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the app framework
- [TensorFlow](https://www.tensorflow.org/) for machine learning capabilities
- [OpenCV](https://opencv.org/) for image processing
