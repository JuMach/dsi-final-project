import speech_recognition as sr
import requests 
import memcache
import logging
import coloredlogs


logger = logging.getLogger(__name__)


def initLogger(level='DEBUG'):
    fmt = '%(asctime)s - %(message)s'
    coloredlogs.install(fmt=fmt, datefmt='%d/%m/%Y %H:%M:%S', level=level)


def speechRecog(r, client, t=4, log=False):
    # r.energy_threshold = 400
    # r.dynamic_energy_threshold = False
    if log: logger.info('Waiting for microphone...', end='', flush=True)

    with sr.Microphone() as source:
        if log: logger.info(' Done', flush=True)
        if log: logger.info("Say something!")
        
        try:
            audio = r.listen(source, timeout=1, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            if log: logger.info("> Wait timeout")
            return

        text = '<>'
        try:
            text = r.recognize_google(audio, language="es-ES")
            if log: logger.info(text)
        except sr.UnknownValueError:
            if log: logger.info("> Could not recognize anything")
            return
        except sr.WaitTimeoutError:
            if log: logger.info("> Wait timeout")
            return

        client.set("speech", text, time=t)


def launch():
    client = memcache.Client([('127.0.0.1', 11211)])
    r = sr.Recognizer()

    speechRecog(r, client)


def continuousSpeech():
    while(True):
        launch()


if __name__ == "__main__":
    while(True):
        launch()
