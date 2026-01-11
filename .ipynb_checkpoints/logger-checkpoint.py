import logging
import sys

def setup_logger(logfile='bot.log'):
    logger = logging.getLogger('stockbot')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Хэндлер для файла
    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Хэндлер для консоли (stdout, чтобы видеть в ноутбуке)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

logger = setup_logger()