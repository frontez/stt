# Speech-to-Text Web App

Welcome to the Speech-to-Text (STT) web application, a simple and efficient tool for converting spoken words into text using the Whisper model by OpenAI. This application leverages NVIDIA's CUDA for GPU acceleration to provide fast and accurate transcriptions.

## Features

- Real-time speech recognition using the Whisper model by OpenAI
- Utilizes NVIDIA's CUDA for fast processing
- Built on Flask, a lightweight web framework
- Offers a simple web interface for audio input
- Supports multiple audio input devices

## Prerequisites

Before running the app, ensure you have the following prerequisites configured:

- [Docker](https://www.docker.com/)
- [NVIDIA Docker Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support
- A CUDA-capable NVIDIA GPU
- Python 3.9

## Getting Started

Follow these instructions to set up and run the application locally.

### Clone the Repository

```bash
git clone https://github.com/frontez/stt.git
cd stt
```

### Build the Docker Image

Build the Docker image using the provided Dockerfile:

```bash
docker build -t stt-app .
```

### Run the Application

Run the Docker container with GPU support:

```bash
docker run --gpus all -p 5000:5000 stt-app
```

This command will start the application on port 5000.

### Access the Web Interface

Open a web browser and navigate to `http://localhost:5000` to access the application.

## How It Works

1. **Audio Input**: Choose an audio input device and click "Start Recording." The app will record audio until you click "Stop Recording."

2. **Transcription**: The app sends the recorded audio to the backend, where it is processed using ffmpeg and transcribed using the Whisper model.

3. **Output**: The transcribed text is displayed on the web interface.

## File Structure

- **app.py**: The main Flask application handling audio processing and transcription requests.
- **requirements.txt**: Lists Python dependencies needed to run the app.
- **Dockerfile**: Specifies the environment setup to run the app in a container.
- **static/index.html**: The user interface for selecting audio sources and displaying transcription results.

## Troubleshooting

- Ensure that you have Docker and NVIDIA Docker Toolkit correctly installed to use the GPU.
- Verify that your audio input devices are correctly configured and accessible by your browser.
- Check the Flask logs for any errors or warnings during processing.

## Contributing

Feel free to contribute to this project by forking the repository and submitting pull requests. Ensure to follow standard coding practices and include clear commit messages.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

This ReadMe file provides a quick overview and step-by-step guide to help users set up and use the Speech-to-Text application seamlessly, powered by the Whisper model for enhanced transcription quality.