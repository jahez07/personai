import google.generativeai as genai 
from elevenlabs import play
# pip3 install -U elevenlabs
from elevenlabs.client import ElevenLabs
# pip3 install faster-whisper
from faster_whisper import WhisperModel
import pyaudio
import os

whisper_size = 'tiny'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device= 'cpu',
    compute_type= 'int_8',
    cpu_threads= num_cores,
    num_workers=num_cores
)


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

model = genai.GenerativeModel('gemini-1.0-pro-latest', generation_config=generation_config)

input_ = '''Greet me when I get home, use my name, "JAHEZ" be polite but funny too, 
            how would you do it if you were assistant ? 
            you do not need to add all the 'I would ...' 
            just give me the direct answer
            As a voice assistant, use short sentences and directly respond to the prompt without excessive information.
            You are expected to be a little funny but prioritizing logic.
            You should use words with care. '''

system_message = '''INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
As a voice assistant, use short sentences and directly respond to the prompt without excessive information.
You are expected to be a little funny but prioritizing logic.
You should use words with care.'''

#response = model.generate_content(input('Ask Gemini: '))
response = model.generate_content(input_)
text = response._result.candidates[0].content.parts[0].text

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join()

audio = client.generate(
  text=text,
  voice="Jessie",
  model="eleven_multilingual_v2"
)
play(audio)