from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
from typing import (
	Optional as _Optional,
	ClassVar as _ClassVar,
	Tuple as _Tuple,
	Dict as _Dict,
)
from helpers.plotting import plot_dendrogram as _plot_dendrogram
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

def question_three():
	cereal = CerealRanking()
	cerealCN = CleanNormalCereal(cereal=cereal)
	
	return