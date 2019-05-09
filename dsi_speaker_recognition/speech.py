import speech_recognition as sr
import requests 
import memcache
import logging
import coloredlogs
import os

logger = logging.getLogger(__name__)


def initLogger(level='DEBUG'):
    fmt = '%(asctime)s - %(message)s'
    coloredlogs.install(fmt=fmt, datefmt='%d/%m/%Y %H:%M:%S', level=level)


def speechRecog(audio, r, client, t=15, log=False):
    text = '<>'
    try:
        text = r.recognize_google(audio, language="es-ES")
        logger.info(text)
    except sr.UnknownValueError:
        logger.info("> Could not recognize anything")
        return
    except sr.WaitTimeoutError:
        logger.info("> Wait timeout")
        return

    client.set("speech", text, time=t)


def speakerRecog(audio, gmm_files):

    #Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                in gmm_files]
    
    sr, audio = read(source + path)
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))


def launch(r, client, modelpath, gmm_files, log=True):
    # r.energy_threshold = 400
    # r.dynamic_energy_threshold = False
    logger.info('Waiting for microphone...')

    with sr.Microphone() as source:
        logger.info(' Done')
        logger.info("Say something!")
        
        try:
            audio = r.listen(source, timeout=1, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            logger.info("> Wait timeout")
            return
    
        speechRecog(audio, r, client, log)
    # print(audio.getframerate())


def continuousSpeech():
    # speech instantiations
    client = memcache.Client([('127.0.0.1', 11211)])
    r = sr.Recognizer()

    # speaker models
    modelpath = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'speaker_models')
    gmm_files = [os.path.join(modelpath, fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]

    while(True):
        launch(r, client, modelpath, gmm_files)


if __name__ == "__main__":
    initLogger()
    continuousSpeech()
