import speech_recognition as sr
import os
import google.generativeai as genai 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from elevenlabs import play
# pip3 install -U elevenlabs
from elevenlabs.client import ElevenLabs
# pip3 install faster-whisper
from faster_whisper import WhisperModel
import pyaudio
import time

# for transcribing the audio to text
whisper_size = 'tiny'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device= 'cpu',
    compute_type= 'int8',
    cpu_threads= num_cores,
    num_workers=num_cores
)

# elevanlabs - api key
client = ElevenLabs(
  api_key="sk_075b6cf0e67280bbe3911ec1def5090c60f41c298fc180fa", # Defaults to ELEVEN_API_KEY
)

GOOGLE_API_KEY = "AIzaSyAlrAFt3wYkTaSj2eOYs0NbiAN12FwXrPk"
genai.configure(api_key=GOOGLE_API_KEY)

# configuring the gemini model

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens":2048,
}
