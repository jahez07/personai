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