from defs import app_logger_name
from datetime import datetime
import platform
import os
import logging
from pathlib import Path
logger = logging.getLogger(app_logger_name)
logger.setLevel(logging.DEBUG)
LOG_FORMATTER = logging.Formatter(fmt=r'%(levelname)s;%(name)s;%(asctime)s %(message)s')
def _setup_handler(handler:logging.Handler):
	handler.setFormatter(LOG_FORMATTER)
	logger.addHandler(handler)
date_format:str = r'%Y-%m-%d %H:%M:%S' if platform.system().casefold() != 'windows' else r'%Y-%m-%d %H_%M_%S'
log_folder=Path('logs')
log_folder.mkdir(mode=0o777, parents=True,exist_ok=True)
FILE_HANDLER = logging.FileHandler(
	filename=f'{str(log_folder)}{os.sep}{datetime.now().strftime(date_format)} {app_logger_name}.log',mode='w'
)
FILE_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)

_setup_handler(FILE_HANDLER)
_setup_handler(STREAM_HANDLER)

RUN_Q1:bool = False
RUN_Q2:bool = False
RUN_Q3:bool = False
logger.debug(f'RUN_Q1:{RUN_Q1}')
logger.debug(f'RUN_Q2:{RUN_Q2}')
logger.debug(f'RUN_Q3:{RUN_Q3}')
try:
	if RUN_Q1:
		from grocery_recommendation.dataset import question_one
		question_one()
		
	if RUN_Q2:
		from university_rankings.dataset import question_two
		question_two()

	if RUN_Q3:
		from cereal.cereal import question_three
		question_three()
except Exception as e:
	logger.error('UNHANDLED EXCEPTION', exc_info=e)

logger.info('exit()')
exit()