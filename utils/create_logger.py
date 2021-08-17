import logging

# https://towardsdatascience.com/how-to-setup-logging-for-your-python-notebooks-in-under-2-minutes-2a7ac88d723d


def create_logger(file,
                  name=__name__,
                  level=logging.DEBUG, 
                  fileHandler_level=logging.DEBUG, 
                  consoleHandler_level=logging.DEBUG):
    # create logger
    logger = logging.getLogger(__name__)
    # set log level for all handlers to debug
    logger.setLevel(level)

    # create console handler and set level to debug
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(consoleHandler_level)

    # create file handler and set level to debug
    fileHandler = logging.FileHandler(file)
    fileHandler.setLevel(fileHandler_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to handlers
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    
    return logger