import logging
import coloredlogs
import os
from multiprocessing import Process
from dsi_facial_recognition import recognizer as recon
# from dsi_facial_recognition import trainer


logger = logging.getLogger(__name__)


def initLogger(level='DEBUG'):
    fmt = '%(asctime)s - %(message)s'
    coloredlogs.install(fmt=fmt, datefmt='%d/%m/%Y %H:%M:%S', level=level)
    # logger.info('Logger is active')

def startRecognizer():
    r.launch()

if __name__ == "__main__":
    initLogger()
    basedir = os.path.abspath(os.path.dirname(__file__))

    os.startfile(os.path.join(basedir, os.path.join("memcached", "memcached.exe")))

    r = recon.Recognizer()
    r.launch()
