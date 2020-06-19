
import logging as standard_logging

standard_logging.basicConfig(level=standard_logging.INFO)
standard_logger = standard_logging.getLogger()


class CustomLogger(object):

    def info(self, param):
        standard_logger.info(param)

    def debug(self, param):
        standard_logger.debug(param)

    def warning(self, param):
        standard_logger.warning(param)

    # Add a method to the logger object (not to the class definition) during runtime x
    def vinfo(self, some_str, some_var):
        assert type(some_str) is str
        if some_var is None:
            some_var = 'None'
        standard_logger.info(some_str + ': ' + str(some_var))


logger = CustomLogger()

