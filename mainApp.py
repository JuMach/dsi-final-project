import logging
import coloredlogs
import os
import sys
from multiprocessing import Process
from dsi_facial_recognition import recognizer as recon
from dsi_speaker_recognition import speech
# from dsi_facial_recognition import trainer


logger = logging.getLogger(__name__)


def initLogger(level='DEBUG'):
    fmt = '%(asctime)s - %(message)s'
    coloredlogs.install(fmt=fmt, datefmt='%d/%m/%Y %H:%M:%S', level=level)
    # logger.info('Logger is active')


def startSpeechRecognizer():
    speech.continuousSpeech()


if __name__ == "__main__":
    initLogger()
    basedir = os.path.abspath(os.path.dirname(__file__))

    os.startfile(os.path.join(basedir, os.path.join("memcached", "memcached.exe")))

    p = Process(target=startSpeechRecognizer)
    p.daemon = True
    p.start()

    r = recon.Recognizer()
    r.launch()
