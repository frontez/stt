import logging
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import whisper
import io
import wave
import ffmpeg
import torch

app = Flask(__name__, static_folder='static')

# Set up logging to output to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Set the Flask app's logger to also handle DEBUG level messages
app.logger.setLevel(logging.DEBUG)

# Load the Whisper model
model = whisper.load_model("base", device="cuda")

def process_audio(audio_data):
    app.logger.info("Audio data received for processing")
    # Save the audio data for debugging purposes
    with open('received_audio.webm', 'wb') as f:
        f.write(audio_data)

    try:
        out_stream, err = (
            ffmpeg
            .input('pipe:0', format='webm')
            .output('pipe:1', format='wav', ar='16000')
            .run(input=audio_data, capture_stdout=True, capture_stderr=True)
        )
        app.logger.info("FFmpeg processing succeeded")
    except ffmpeg.Error as e:
        app.logger.error(f'FFmpeg failed with error: {e.stderr.decode()}')
        raise ValueError(f'FFmpeg failed with error: {e.stderr.decode()}')

    wave_file = wave.open(io.BytesIO(out_stream), "rb")
    n_frames = wave_file.getnframes()
    audio_content = wave_file.readframes(n_frames)
    audio_array = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32) / 32768.0

    result = model.transcribe(audio_array)
    # Use string formatting correction here
    app.logger.info(f"Transcription result: {result['text']}")
    return result['text']

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder,'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_data = request.data
    try:
        text = process_audio(audio_data)
        return jsonify({'text': text})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

with app.app_context():
    app.logger.info(f"CUDA is available: {torch.cuda.is_available()}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)