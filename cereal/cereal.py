from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
from typing import (
	List as _List,
	Optional as _Optional,
	ClassVar as _ClassVar,
	Tuple as _Tuple,
	Dict as _Dict,
	Literal as _Literal,
)
from helpers.plotting import (
	CustomDendrogram as _CustomDendrogram,
	generate_many_dendrograms as _generate_many_dendrograms
)
import pandas as _pd
from helpers.dataset import DatasetBase as _DatasetBase
from sklearn.preprocessing import Normalizer as _Normalizer
from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram as _dendrogram
import numpy as _np
from defs import QuestionThreeData as _Q3D

from helpers.dataset import (
	DatasetBase as _DatasetBase,
	DatasetCSV as _DatasetCSV,
	DatasetCSVReadOnly as _DatasetCSVReadOnly,
	DatasetSaveMixin as _DatasetSaveMixin,
)
import logging
logger = logging.getLogger(_Q3D.logger_name)
_Q3D.folder_dendrograms.mkdir(mode=0o775, parents=True,exist_ok=True)
import matplotlib.pyplot as _plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes

class CerealRanking(_DatasetCSVReadOnly):
	default_path:_ClassVar[_Path] = _Q3D.Cereal
	logger.debug(f'CerealRanking.default_path: {default_path}')

	def __init__(self, path:_Path=default_path):
		logger.debug('CerealRanking.__init__')
		super().__init__(path=path, frame=None)
		self.load()
		
	class Columns(_StrEnum):
		Name = 'name'
		Mfr = 'mfr'
		Type = 'type'
		Calories = 'calories'
		Protein = 'protein'
		Fat = 'fat'
		Sodium = 'sodium'
		Fiber = 'fiber'
		Carbo = 'carbo'
		Sugars = 'sugars'
		Potass = 'potass'
		Vitamins = 'vitamins'
		Shelf = 'shelf'
		Weight = 'weight'
		Cups = 'cups'
		Rating = 'rating'

class CleanNormalCereal(_DatasetCSV):
	default_path:_ClassVar[_Path] = _Q3D.CerealCleanedNormalized
	logger.debug(f'CleanNormalUniversity.default_path: {default_path}')
	def __init__(self, cereal:_Optional[CerealRanking]=None):
		logger.debug(f'CleanNormalUniversity.__init__(rankings={cereal})')
		super().__init__(path=self.default_path, frame=None)
		if self.path is not None and self.path.exists():
			logger.debug('CleanNormalUniversity.__init__ - loaded from file')
			self.load()
		elif cereal is not None:
			logger.debug('CleanNormalUniversity.__init__ - cleaning & normalizing from dataset')
			frame:_pd.DataFrame = cereal.get_frame().drop(columns=[
				CerealRanking.Columns.Name,
				CerealRanking.Columns.Mfr,
				CerealRanking.Columns.Type,
			]).dropna(how='any', axis=0)
			normalizer = _Normalizer()
			self.frame = _pd.DataFrame(normalizer.fit_transform(frame), columns=frame.columns)
		else:
			raise Exception('invalid path, rankings = None')

class CleanNormalLabeled(_DatasetBase, _DatasetSaveMixin):
	COLUMN_LABEL:_ClassVar[str] = 'LABEL'
	default_path:_ClassVar[_Path] = _Q3D.CerealCleanedNormalLabeled
	dendrogram:_CustomDendrogram

	def __init__(self, cleanCereal:CleanNormalCereal, dendrogram:_CustomDendrogram):
		logger.debug('CleanNormalLabeled.__init__')
		super().__init__(
			path=self.default_path,
			frame=cleanCereal.get_frame().copy()
		)
		self.dendrogram = dendrogram
		self.get_frame()[self.COLUMN_LABEL] = dendrogram.get_model().labels_

	def get_summary_statistics(self)->_Tuple[_pd.DataFrame, _pd.DataFrame]:
			''' Calculates mean & median with groupby=LABEL, saves and returns frames. '''
			frame_by_label = self.get_frame().groupby(by=self.COLUMN_LABEL)
			median = frame_by_label.median()
			mean = frame_by_label.mean()
			return median,mean
	
def question_three():
	cereal = CerealRanking()
	cerealCN = CleanNormalCereal(cereal=cereal)
	complete_distance_thresholds=[x/100 for x in range(30, 105, 5)]
	single_distance_thresholds=[x/100 for x in range(4,32,2)]
	dendrograms = _generate_many_dendrograms(
		data=cerealCN.get_frame(),
		linkage='single',
		thresholds=single_distance_thresholds,
		
		show=False,
		folder=_Q3D.folder_dendrograms,
		file_str_fmt= 'linkage={linkage} dt={distance_threshold}.tiff',
		save_to_file=True, clobber=False, raise_err_exists=False
	)
	logger.info('Generated single linkeage dendrograms.')
	dendrograms.extend(_generate_many_dendrograms(
		data=cerealCN.get_frame(),
		linkage='complete',
		thresholds=complete_distance_thresholds,
		
		show=False,
		folder=_Q3D.folder_dendrograms,
		file_str_fmt= 'linkage={linkage} dt={distance_threshold:0.2f}.tiff',
		save_to_file=True, clobber=False, raise_err_exists=False

	))
	logger.info(f'Generated complete linkeage dendrograms. ({len(dendrograms)})')
	labeled = CleanNormalLabeled(cleanCereal=cerealCN, dendrogram=_CustomDendrogram(
		data=cerealCN.get_frame(),
		distance_threshold=0.80,
		linkage='complete',
		file=_Path(''), save_to_file=False,
		show=False,
	))
	labeled.save()
	median,mean = labeled.get_summary_statistics()
	
	median.to_csv(_Q3D.folder_datasets.joinpath(
		labeled.dendrogram.format_from_vars('CCNL Group by Label linkage={linkage} dt={distance_threshold} median.csv')
	))
	median.to_csv(_Q3D.folder_datasets.joinpath(
		labeled.dendrogram.format_from_vars('CCNL Group by Label linkage={linkage} dt={distance_threshold} mean.csv')
	))
	return