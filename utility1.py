import logging
import os
import sys
from importlib import reload
from datetime import datetime
reload(sys)


timestamp_format = "%d/%m/%Y %H:%M:%S.%f"
start_time = "Start time: "
end_time = "End time: "
response_data_key = "response_data"
error_key = "errors"
request_header_delimiter = "$"


#class_name, method_name = "utility.py", <function_name>.__name__


cwd = os.getcwd()  # current working directory path

# Setting path for creating log files
log_dir=os.path.join(cwd+os.sep+"logs"+os.sep)

# Setting Log configuration
formatter = logging.Formatter("[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
logfile=log_dir+ "sklearn_pipeline.log"


# logging.basicConfig(level = logging.INFO, filename = logfile)
#
# logging.basicConfig(level = logging.INFO, filename = logfile, filemode = 'w')

logging.basicConfig(format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
                    filename = logfile, level = logging.DEBUG)

console = logging.StreamHandler()
# add the handler to the root logger
logger=logging.getLogger(' ').addHandler(console)


def logger_info(class_name, method_name, info_msg):

    logging.info(str(class_name) + ' : ' + str(method_name) + ' : ' + str(info_msg))


def logger_debug(class_name, method_name, debug_msg):

    logging.debug(str(class_name) + ' : ' + str(method_name) + ' : ' + str(debug_msg))


def logger_error(class_name, method_name, error_msg):

    logging.error(str(class_name) + ' : ' + str(method_name) + ' : ' + str(error_msg))


def logger_start(class_name, method_name):
    curr_time = datetime.now()
    formatted_time = curr_time.strftime(timestamp_format)
    info_msg = start_time + str(formatted_time)
    logging.info(str(class_name) + ' : ' + str(method_name) +' : ' + str(info_msg))


def logger_end(class_name, method_name):
    curr_time = datetime.now()
    formatted_time = curr_time.strftime(timestamp_format)
    info_msg = end_time + str(formatted_time)
    logging.info(str(class_name) + ' : ' + str(method_name) + ' : ' + str(info_msg))





