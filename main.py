# pip3 install SpeechRecognition
import speech_recognition as sr
import google.generativeai as genai 
from elevenlabs import play
# pip3 install -U elevenlabs
from elevenlabs.client import ElevenLabs
# pip3 install faster-whisper
from faster_whisper import WhisperModel
import pyaudio
import os
import time

wake_word = 'personai'
listening_for_wake_word = True

whisper_size = 'tiny'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device= 'cpu',
    compute_type= 'int8',
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

convo = model.start_chat()

input_ = ''' '''

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

system_message = system_message.replace(f'\n','')
convo.send_message(system_message)
#response = model.generate_content(input('Ask Gemini: '))
#response = model.generate_content(input_)
#text = response._result.candidates[0].content.parts[0].text


# to allow the system to recognise the audio
r = sr.Recognizer()
source = sr.Microphone()

# to convert audio to text
def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text_ = ''.join(segment.text for segment in segments)
    return text_


# generate the audio of the text passed
def audio_gen(text):
    audio_ = client.generate(
        text=text,
        voice="Jessie",
        model="eleven_multilingual_v2"
    )
    return audio_


# wake word
def listen_for_wake_word(audio):
    global listening_for_wake_word

    wake_audio_path = 'wake_detect.wav'
    with open(wake_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    text_input = wav_to_text(wake_audio_path)
    if wake_word in text_input.lower().strip():
        print('Wake word detected. Please speak your prompt to Personai')
        listening_for_wake_word = False

def prompt_gpt(audio):
    global listening_for_wake_word

    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        
        # converting the audio(prompt) from user to 
        prompt_text = wav_to_text(prompt_audio_path)

        # checking if the prompt is empty
        if len(prompt_text.strip()) == 0:
            text = "sorry, could not get you!"
            audio = audio_gen(text)
            play(audio)
            listening_for_wake_word = True
        else:
            print('User' + prompt_text)
            convo.send_message(prompt_text)
            text = convo.last.text

            print('Personai: ', text)
            audio = audio_gen(text)
            play(audio)

            print('\nSay', wake_word, 'to wake me up.\n')
            listening_for_wake_word = True
    except Exception as e:
        print('Prompt error: ', e)

def callback(recognizer, audio):
    global listening_for_wake_word

    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    print('\nSay', wake_word, 'to wake me up.\n')

    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)


if __name__ == '__main__':
    start_listening()