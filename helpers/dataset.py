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
from defs import app_logger_name
import logging
logger = logging.getLogger(f'{app_logger_name}.helpers.dataset')

class _DatasetAttributesProtocol(_Protocol):
	''' This is not a 'real' class, but defines the expected substructure of dataset base.
	This is for mixin classes.
	'''
	path:_Optional[_Path]
	frame:_Optional[_pd.DataFrame]

class DatasetSaveMixin(_DatasetAttributesProtocol):
	def save(self, clobber:bool=True, include_index:bool=False):
		''' Saves the dataset to the file. '''
		logger.debug(f'DatasetSaveMixin.save(clobber={clobber}, include_index={include_index})')
		if self.frame is None:
			err = _DatasetMissingFrame('Cannot save frame, frame is not loaded.')
			logger.error('DatasetSaveMixin.save', exc_info=err)
			raise err
		if self.path is None:
			err = _DatasetInvalidPath('Cannot save frame, Path is None.')
			logger.error('DatasetSaveMixin.save', exc_info=err)
			raise err
		if (not clobber) and self.path.exists():
			err = _DatasetFileExists('Cannot save frame, file already exists.')
			logger.error('DatasetSaveMixin.save', exc_info=err)
			raise err
		logger.debug('DatasetSaveMixin.save - saving')
		self.frame.to_csv(
			path_or_buf=self.path,
			index=include_index,
		)

class DatasetLoadCSVMixin(_DatasetAttributesProtocol):
	def load(self)->None:
		''' Load the dataset from the path. '''
		logger.debug('DatasetLoadCSVMixin.load()')
		if self.path is None:
			err =  _DatasetInvalidPath('Cannot load frame without path.')
			logger.error('DatasetSaveMixin.load', exc_info=err)
			raise err
		logger.debug(f'DatasetLoadCSVMixin.load - loading from {self.path}')
		self.frame = _pd.read_csv(
			filepath_or_buffer = self.path,
		)

class DatasetTextFileMixin(_DatasetAttributesProtocol):
	def load(self)->None:
		''' Every line in a file is loaded into an Nx1 Frame'''
		logger.debug('DatasetTextFileMixin.load()')
		if self.path is None:
			err = _DatasetInvalidPath('Cannot load frame without path.')
			logger.error('DatasetTextFileMixin.load', exc_info=err)
			raise err
		logger.debug(f'DatasetTextFileMixin.load - loading from {self.path}')
		with open(self.path, 'r') as file:
			records = [(line.strip(),) for  line in file]
		logger.debug('DatasetTextFileMixin.load - finished reading file.')
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
		logger.debug(f'DatasetBase.__init__(path={path}, frame={frame})')
		self.path = path
		self.frame = frame

	def get_frame(self)->_pd.DataFrame:
		''' Returns the frame or raises DatasetMissingFrame'''
		logger.debug('DatasetBase.get_frame()')
		if self.frame is None:
			err = _DatasetMissingFrame()
			logger.error('DatasetBase.get_frame',exc_info=err)
			raise err
		return self.frame

class DatasetCSVReadOnly(DatasetBase, DatasetLoadCSVMixin):
	''' A Dataset which can be loaded from CSV'''
	def __init__(self,
			path:_Optional[_Path] = None,
			frame:_Optional[_pd.DataFrame] = None,
			)->None:
		logger.debug(f'DatasetCSVReadOnly.__init__(path={path}, frame={frame})')
		super().__init__(path=path,frame=frame)
	
	@classmethod
	def create_from_file(cls,path:_Path):
		''' Initializes and loads the dataset'''
		logger.debug(f'DatasetCSVReadOnly.create_from_file(path={path}) - cls.__name__={cls.__name__}')
		if not path.exists():
			err = FileNotFoundError('Cannot load dataset from file. File Not Found.')
			logger.error('DatasetCSVReadOnly.create_from_file',exc_info=err)
			raise err
		dataset = cls(path=path)
		dataset.load()
		return dataset
	
class DatasetCSV(DatasetCSVReadOnly, DatasetSaveMixin):
	''' A Dataset which can be loaded & saved to CSV'''
	def __init__(self,
			path:_Optional[_Path] = None,
			frame:_Optional[_pd.DataFrame] = None,
			)->None:
		logger.debug(f'DatasetCSV.__init__(path={path}, frame={frame})')
		super().__init__(path=path,frame=frame)
	