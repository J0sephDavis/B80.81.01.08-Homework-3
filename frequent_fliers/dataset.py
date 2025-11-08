from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
from typing import (
	Optional as _Optional,
	ClassVar as _ClassVar,
	Tuple as _Tuple,
)
from helpers.plotting import (
	CustomDendrogram as _CustomDendrogram,
)
import pandas as _pd
from helpers.dataset import DatasetBase as _DatasetBase
from sklearn.preprocessing import Normalizer as _Normalizer
from defs import FlierData as _FD

from helpers.dataset import (
	DatasetBase as _DatasetBase,
	DatasetCSV as _DatasetCSV,
	DatasetCSVReadOnly as _DatasetCSVReadOnly,
	DatasetSaveMixin as _DatasetSaveMixin,
)
import logging
logger = logging.getLogger(_FD.logger_name)

class FrequentFliers(_DatasetCSVReadOnly):
	default_path:_ClassVar[_Path] = _FD.FrequentFliers
	log_prefix:_ClassVar[str] = f'{__qualname__}'
	logger.debug(f'{log_prefix}.default_path: {__qualname__}')

	def __init__(self, path:_Path=default_path):
		logger.debug(f'{self.log_prefix}.__init__')
		super().__init__(path=path, frame=None)
		self.load()
	
	class Columns(_StrEnum):
		ID = 'ID#'
		Balance = 'Balance'
		Qual_miles = 'Qual_miles'
		cc1_miles = 'cc1_miles'
		cc2_miles = 'cc2_miles'
		cc3_miles = 'cc3_miles'
		Bonus_miles = 'Bonus_miles'
		Bonus_trans = 'Bonus_trans'
		Flight_miles_12mo = 'Flight_miles_12mo'
		Flight_trans_12 = 'Flight_trans_12'
		Days_since_enroll = 'Days_since_enroll'
		Award = 'Award?'

class FrequentFliersNormalized(_DatasetCSV):
	default_path:_ClassVar[_Path] = _FD.Normalized
	log_prefix:_ClassVar[str] = f'{__qualname__}'
	logger.debug(f'{log_prefix}.default_path: {__qualname__}')
	
	def __init__(self, frequent_fliers:_Optional[FrequentFliers]=None, frame:_Optional[_pd.DataFrame] = None):
		logger.debug(f'{self.log_prefix}.__init__({frequent_fliers})')
		super().__init__(path=self.default_path, frame=frame)
		if self.frame is not None:
			logger.debug(f'{self.log_prefix} loaded from frame.')
			self.frame=frame
		elif self.path is not None and self.path.exists():
			logger.debug(f'{self.log_prefix}.__init__ - loaded from file')
			self.load()
		elif frequent_fliers is not None:
			self._clean_and_normalize(frequent_fliers)
		else:
			raise Exception('invalid path, rankings = None')
	
	def _clean_and_normalize(self,frequent_fliers:FrequentFliers):
		logger.debug(f'{self.log_prefix}._clean_and_normalize')
		frame:_pd.DataFrame = frequent_fliers.get_frame().drop(columns=[
			FrequentFliers.Columns.ID,
			FrequentFliers.Columns.Award, # Categorical
		]).dropna(how='any', axis=0)
		normalizer = _Normalizer()
		self.frame = _pd.DataFrame(normalizer.fit_transform(frame), columns=frame.columns)

	def sample_existing(self, sample_frac:float)->'FrequentFliersNormalized':
		frame = self.get_frame().sample(frac=sample_frac).copy()
		ffn = FrequentFliersNormalized()
		return ffn

class FrequentFliersLabeled(_DatasetBase, _DatasetSaveMixin):
	COLUMN_LABEL:_ClassVar[str] = 'LABEL'
	dendrogram:_CustomDendrogram
	log_prefix:_ClassVar[str] = f'{__qualname__}'

	def __init__(self,
			  ffNorm:FrequentFliersNormalized,
			  dendrogram:_CustomDendrogram,
			  file_path:_Path,
			):
		logger.debug(f'{self.log_prefix}.__init__')
		super().__init__(
			path=file_path,
			frame=ffNorm.get_frame().copy()
		)
		self.dendrogram = dendrogram
		self.get_frame()[self.COLUMN_LABEL] = dendrogram.get_model().labels_

	def get_summary_statistics(self)->_Tuple[_pd.DataFrame, _pd.DataFrame]:
			''' Calculates mean & median with groupby=LABEL, saves and returns frames. '''
			frame_by_label = self.get_frame().groupby(by=self.COLUMN_LABEL)
			median = frame_by_label.median()
			mean = frame_by_label.mean()
			return median,mean