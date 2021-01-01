import logging
import sys

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    shandler = logging.StreamHandler(sys.stdout)
    shandler.setFormatter(formatter)
    fhandler = logging.FileHandler(f'logs/{name}.log')
    fhandler.setFormatter(formatter)
    logger.addHandler(shandler)
    logger.addHandler(fhandler)
    return logger
