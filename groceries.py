from enum import StrEnum as _StrEnum
import pandas as _pd
from helpers.dataset import DatasetBase as _DatasetBase
from defs import FILEPATH_GROCERIES as _FILEPATH_GROCERIES
from helpers.exceptions import DatasetInvalidPath as _DatasetInvalidPath

class GroceriesDataset(_DatasetBase):
	''' Class representing the original groceries dataset'''
	def __init__(self)->None:
		super().__init__(_FILEPATH_GROCERIES)

	def load(self, force_reload: bool = False, **read_csv_kwargs) -> None:
		''' We override the origina load method because this data is not rectangular. '''
		if (not force_reload) and (self.frame is not None):
			return
		if self.path is None:
			raise _DatasetInvalidPath('Cannot load frame, Path is None.')
		if not self.path.exists():
			raise FileNotFoundError('Cannot load frame, Dataset file not found.')
		# ----
		with open(self.path, 'r') as file:
			records = [(line.strip(),) for  line in file]
		self.frame = _pd.DataFrame.from_records(data=records, columns=['Transaction'])
		return
	
	def save(self, clobber:bool=True, include_index:bool=False, **to_csv_kwargs)->None:
		''' You cannot overwrite the original dataset. '''
		raise Exception('Forbidden')
	
	def hot_encode_transactions(self)->_pd.DataFrame: # QUESTION 1.A
		''' Used to hotencode transactions when making the recommendation dataset. '''
		frame = self.get_frame()
		dumm = frame['Transaction'].str.get_dummies(sep=',').astype(bool)
		return dumm