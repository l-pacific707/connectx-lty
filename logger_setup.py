# logger_setup.py
import logging

def get_logger(name="AlphaZeroTraining", log_file="AlphaZeroTraining.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        # 파일 핸들러
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=500*1024*1024, backupCount=10)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
