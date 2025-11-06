from pathlib import (
	Path as _Path
)
import pandas as _pd
from typing import (
	Optional as _Optional,
	List as _List,
	Tuple as _Tuple,
	Union as _Union,
	Protocol as _Protocol,
)
from enum import StrEnum as _StrEnum
from .exceptions import (
	DatasetFileExists as _DatasetFileExists,
	DatasetMissingFrame as _DatasetMissingFrame,
	DatasetInvalidPath as _DatasetInvalidPath,
)


class _DatasetAttributesProtocol(_Protocol):
	''' This is not a 'real' class, but defines the expected substructure of dataset base.
	This is for mixin classes.
	'''
	path:_Optional[_Path]
	frame:_Optional[_pd.DataFrame]

class DatasetSaveMixin(_DatasetAttributesProtocol):
	def save(self, clobber:bool=True, include_index:bool=False):
		''' Saves the dataset to the file. '''
		if self.frame is None:
			raise _DatasetMissingFrame('Cannot save frame, frame is not loaded.')
		if self.path is None:
			raise _DatasetInvalidPath('Cannot save frame, Path is None.')
		if (not clobber) and self.path.exists():
			raise _DatasetFileExists('Cannot save frame, file already exists.')
		
		self.frame.to_csv(
			path_or_buf=self.path,
			index=include_index,
		)

class DatasetLoadCSVMixin(_DatasetAttributesProtocol):
	def load(self)->None:
		''' Load the dataset from the path. '''
		if self.path is None:
			raise _DatasetInvalidPath('Cannot load frame without path.')
		print(f'loading dataset from {self.path}')
		self.frame = _pd.read_csv(
			filepath_or_buffer = self.path,
		)

class DatasetTextFileMixin(_DatasetAttributesProtocol):
	def load(self)->None:
		''' Every line in a file is loaded into an Nx1 Frame'''
		if self.path is None:
			raise _DatasetInvalidPath('Cannot load frame without path.')
		print(f'loading dataset from {self.path}')
		with open(self.path, 'r') as file:
			records = [(line.strip(),) for  line in file]
		self.frame = _pd.DataFrame.from_records(data=records, columns=['Transaction'])
		return


class DatasetBase():
	''' Represents a dataframe and file path.'''
	path:_Optional[_Path]
	frame:_Optional[_pd.DataFrame]

	def __init__(self,
			path:_Optional[_Path] = None,
			frame:_Optional[_pd.DataFrame] = None,
			)->None:
		'''
		path -- the path to the dataset
		frame -- If you already have the dataframe, give it here.
		'''
		self.path = path
		self.frame = frame

	def get_frame(self)->_pd.DataFrame:
		''' Returns the frame or raises DatasetMissingFrame'''
		if self.frame is None:
			raise _DatasetMissingFrame()
		return self.frame

class DatasetCSVReadOnly(DatasetBase, DatasetLoadCSVMixin):
	''' A Dataset which can be loaded from CSV'''
	def __init__(self,
			path:_Optional[_Path] = None,
			frame:_Optional[_pd.DataFrame] = None,
			)->None:
		super().__init__(path=path,frame=frame)
	
	@classmethod
	def create_from_file(cls,path:_Path):
		''' Initializes and loads the dataset'''
		if not path.exists():
			raise FileNotFoundError('Cannot load dataset from file. File Not Found.')
		dataset = cls(path=path, frame=None)
		dataset.load()
		return dataset
	
class DatasetCSV(DatasetCSVReadOnly, DatasetSaveMixin):
	''' A Dataset which can be loaded & saved to CSV'''
	def __init__(self,
			path:_Optional[_Path] = None,
			frame:_Optional[_pd.DataFrame] = None,
			)->None:
		super().__init__(path=path,frame=frame)
	