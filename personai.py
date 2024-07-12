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

# this would give more losen response from gemini
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]


model = genai.GenerativeModel('gemini-1.0-pro-latest', generation_config=generation_config)

convo = model.start_chat()

input_ = '''Greet me when I get home, use my name, "JAHEZ" be polite but funny too, 
            how would you do it if you were assistant ? 
            you do not need to add all the 'I would ...' 
            just give me the direct answer
            As a voice assistant, use short sentences and directly respond to the prompt without excessive information.
            You are expected to be a little funny but prioritizing logic.
            You should use words with care.'''

system_message = '''INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
                    to this system message. After the system message respond normally.
                    SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
                    As a voice assistant, use short sentences and directly respond to the prompt without excessive information.
                    You are expected to be a little funny but prioritizing logic.
                    You should use words with care. Greet me when I get home, use my name, "JAHEZ" be polite but funny too, 
                    how would you do it if you were assistant ? 
                    you do not need to add all the 'I would ...' 
                    just give me the direct answer
                    As a voice assistant, use short sentences and directly respond to the prompt without excessive information.
                    You are expected to be a little funny but prioritizing logic.
                    You should use words with care.'''

# generate the audio of the text passed
def audio_gen(text):
    audio_ = client.generate(
        text=text,
        voice="Jessie",
        model="eleven_multilingual_v2"
    )
    return audio_

# to get the response 
def gemini(text):
    convo.send_message(text)
    output = convo.last.text
    audio = audio_gen(output)
    play(audio)
