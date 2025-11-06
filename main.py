from defs import app_logger_name
from datetime import datetime
import logging
logger = logging.getLogger(app_logger_name)
LOG_FORMATTER = logging.Formatter(fmt=r'%(levelname)s;%(name)s;%(asctime)s | %(message)s')
def _setup_handler(handler:logging.Handler):
	handler.setFormatter(LOG_FORMATTER)
	logger.addHandler(handler)
FILE_HANDLER = logging.FileHandler(
	filename=f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {app_logger_name}.log',mode='w'
)
FILE_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)
_setup_handler(FILE_HANDLER)
_setup_handler(STREAM_HANDLER)

from grocery_recommendation.dataset import question_one
question_one()

from university_rankings.dataset import question_two
question_two()
exit()