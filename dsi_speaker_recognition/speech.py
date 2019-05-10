import speech_recognition as sr
import memcache
import logging
import coloredlogs
import io
import os
import time
import pickle
import numpy as np
from dsi_speaker_recognition import speakerfeatures as sf
from scipy.io.wavfile import read as wavRead


logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000


def initLogger(level='DEBUG'):
    fmt = '%(asctime)s - %(message)s'
    coloredlogs.install(fmt=fmt, datefmt='%d/%m/%Y %H:%M:%S', level=level)


def speechRecog(audio, r, client, t=15, log=False):
    text = '<...>'
    try:
        text = r.recognize_google(audio, language="es-ES")
        if log: logger.info(text)
    except sr.UnknownValueError:
        if log: logger.info("> Could not recognize anything")
        return
    except sr.WaitTimeoutError:
        if log: logger.info("> Wait timeout")
        return

    logger.info('Detected speech: "{}"'.format(text))
    client.set("speech", text, time=4)


def speakerRecog(audio, gmm_files, client):
    #Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                in gmm_files]

    vector = sf.extract_features(audio, SAMPLE_RATE)
    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i] 
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    logger.info('Detected speaker: "{}"'.format(speakers[winner]))
    client.set("speaker", speakers[winner], time=4)


def launch(r, client, modelpath, gmm_files, log=False):
    # r.energy_threshold = 400
    # r.dynamic_energy_threshold = False
    if log: logger.info('Waiting for microphone...')

    with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
        if log: logger.info(' Done')
        if log: logger.info("Say something!")

        try:
            audio = r.listen(source, timeout=1, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            if log: logger.info("> Wait timeout")
            return

        speechRecog(audio, r, client, log)

        sampleRate, audioNPArray = wavRead(io.BytesIO(audio.get_wav_data()))
        speakerRecog(audioNPArray, gmm_files, client)


def continuousSpeech():
    initLogger()
    # speech instantiations
    client = memcache.Client([('127.0.0.1', 11211)])
    r = sr.Recognizer()

    # speaker models
    modelpath = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'speaker_models')
    gmm_files = [os.path.join(modelpath, fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]

    while(True):
        duration = time.time()
        launch(r, client, modelpath, gmm_files)

        if time.time() - duration < 2:
            pass
            # print(time.time() - duration)
            # time.sleep(2 - (time.time() - duration))

if __name__ == "__main__":
    continuousSpeech()
