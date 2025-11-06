from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
from typing import (
	Optional as _Optional,
	List as _List,
	Union as _Union,
)
import pandas as _pd
from helpers.dataset import DatasetBase as _DatasetBase
from helpers.exceptions import DatasetInvalidPath as _DatasetInvalidPath
from sklearn.preprocessing import Normalizer as _Normalizer

from defs import (
	FILEPATH_UNIVERSITY_RANKINGS as _FILEPATH_UNIVERSITY_RANKINGS,
)

class UniversityDataset(_DatasetBase):
	def __init__(self,
			path=_FILEPATH_UNIVERSITY_RANKINGS,
			)->None:
		super().__init__(path)
	def save(self, clobber:bool=True, include_index:bool=False, **to_csv_kwargs)->None:
		''' You cannot overwrite the original dataset. '''
		raise Exception('Forbidden')
	
	def get_clean_copy(self)->_pd.DataFrame:
		''' returns a copy of the dataset without any null values (removed entire record)'''
		return self.get_frame().copy().dropna(axis=0, how='any')