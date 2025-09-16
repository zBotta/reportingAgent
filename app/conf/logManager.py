""""
logManager.py
"""

import os
import logging
from conf.projectConfig import Config as cf


class Logger(logging.Logger):
    def __init__(self, name: str) -> None:
        if not os.path.isdir(cf.LOG.LOG_DIR):
            os.makedirs(cf.LOG.LOG_DIR)

        super().__init__(name)
        self.setLevel(cf.LOG.LOG_LEVEL)
        self.propagate = False
        # Adding a console handler
        console_handler = ConsoleHandler()
        self.addHandler(console_handler)

        # Adding a file handler
        file_handler = CustomFileHandler()
        self.addHandler(file_handler)
    
    # setup log functions
    def clear():
        """ Open log file and """
        with open(cf.LOG.LOG_FILE, "w"):
            pass

    # specific log functions

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.CRITICAL):
            self._log(logging.CRITICAL, msg, args, **kwargs)


class ConsoleHandler(logging.StreamHandler):
    def __init__(self, level: int = cf.LOG.CONSOLE_LEVEL) -> None:
        super().__init__()
        formatter = logging.Formatter(cf.LOG.FORMAT, datefmt=cf.LOG.DATE_TIME_FORMAT)
        self.setFormatter(formatter)
        self.setLevel(level)


class CustomFileHandler(logging.FileHandler):
    def __init__(self):
        super().__init__(cf.LOG.LOG_FILE, encoding="UTF-8")
        formatter =  logging.Formatter(cf.LOG.FORMAT, datefmt=cf.LOG.DATE_TIME_FORMAT)
        self.setFormatter(formatter)
        self.setLevel(cf.LOG.LOG_LEVEL)


# class Log:
#     def __init__(self, file_path: str):
#         name = os.path.basename(file_path)
#         self.logger = logging.getLogger(name)
#         logging.basicConfig(level=logging.DEBUG, 
#                             format=cf.LOG.FORMAT, 
#                             datefmt=cf.LOG.DATE_TIME_FORMAT, 
#                             filename=cf.LOG.LOG_FILE, 
#                             filemode='w')

#     def debug(self, msg):
#         if self.logger.isEnabledFor(logging.DEBUG):
#             self.logger.debug(msg=msg)    
        
#     def info(self, msg):
#         if self.logger.isEnabledFor(logging.INFO):
#             self.logger.info(msg=msg)

#     def warning(self, msg):
#         if self.logger.isEnabledFor(logging.WARNING):
#             self.logger.warning(msg=msg)

#     def error(self, msg):
#         if self.logger.isEnabledFor(logging.ERROR):
#             self.logger.error(msg=msg)
        
#     def critical(self, msg):
#         if self.logger.isEnabledFor(logging.CRITICAL):
#             self.logger.critical(msg=msg)
