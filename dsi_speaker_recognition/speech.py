import speech_recognition as sr
import requests 
import memcache

r = sr.Recognizer()
# r.energy_threshold = 400
# r.dynamic_energy_threshold = False
print('Waiting for speaker...')

with sr.Microphone() as source:
    print("Say something")
    print('Starting to record...', end='')
    audio = r.listen(source, timeout=1, phrase_time_limit=3)
    print(' Done', flush=True)

    text = '<>'
    try:
        text = r.recognize_google(audio, language="es-ES")
        print(text)
    except sr.UnknownValueError:
        print ("Could not recognize your voice")

    client = memcache.Client([('127.0.0.1', 11211)])
    client.set("speech", text, time=10)
