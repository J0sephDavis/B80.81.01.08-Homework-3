from enum import StrEnum as _StrEnum
from pathlib import Path
from typing import (
	Optional as _Optional,
	List as _List,
)
import pandas as _pd
from helpers.dataset import DatasetBase as _DatasetBase
from helpers.exceptions import DatasetInvalidPath as _DatasetInvalidPath

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