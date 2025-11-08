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
	default_Path:_ClassVar[_Path] = _Q3D.CerealCleanedNormalLabeled
	aggcluster_default_args:_ClassVar[_Dict] = {
		'metric':'euclidean',
		'compute_full_tree':True,
		'compute_distances':True,
		'n_clusters':None, # must be none when distance_threshold is set.
	}
	distance_threshold:float
	model:_AgglomerativeClustering
	linkage:_Literal['single','complete']

	def __init__(self, cleanCereal:CleanNormalCereal, distance_threshold:float, linkage:_Literal['single','complete']):
		logger.debug('CleanNormalLabeled.__init__')
		super().__init__(path=self.default_Path, frame=None)
		self.distance_threshold = distance_threshold
		self.linkage = linkage
		self.figure_file = _Q3D.folder_dendrograms.joinpath(f'dendrogram linkage={self.linkage} dt={self.distance_threshold:0.2f}.tiff')
		
		self.model = _AgglomerativeClustering(
			distance_threshold=self.distance_threshold,
			linkage=linkage,
			**self.aggcluster_default_args
		)
		self.frame = cleanCereal.get_frame().copy()
		self.model.fit(self.frame)
		self.get_frame()[self.COLUMN_LABEL]=self.model.labels_
	
	def plotsave_dendrogram(self, save_to_file:bool=True, show:bool=False)->_Tuple[_Figure,_Axes]:
		logger.debug(f'plot_and_save_dendrogram({self.figure_file},...,...)')
		return _plot_dendrogram(
			file=self.figure_file,
			distance_threshold=self.distance_threshold,
			model=self.model,
			save_to_file=save_to_file,
			show=show,
			linkage=self.linkage
		)

def generate_many_dendrograms(cleanData:CleanNormalCereal, save_to_file:bool, show:bool, linkage, thresholds:_List[float])->None:
	logger.info('cereal.generate_many_dendrograms')
	CleanNormalLabeled(cleanData, 0, linkage).plotsave_dendrogram(save_to_file=save_to_file,show=show)
	for it in thresholds:
		if it==0:# skip
			continue
		logger.debug(f'distance_threshold:{it}')
		data = CleanNormalLabeled(cleanData,it, linkage)
		if data.figure_file.exists() and save_to_file:
			logger.info(f'{data.figure_file} already exists. skipping...')
			continue
		data.plotsave_dendrogram(save_to_file=save_to_file,show=show)


def question_three():
	cereal = CerealRanking()
	cerealCN = CleanNormalCereal(cereal=cereal)
	complete_distance_thresholds=[x/100 for x in range(30, 105, 5)]
	single_distance_thresholds=[x/100 for x in range(4,32,2)]
	_generate_many_dendrograms(
		data=cerealCN.get_frame(),
		linkage='single',
		thresholds=single_distance_thresholds,
		
		show=False,
		folder=_Q3D.folder_dendrograms,
		file_str_fmt= 'linkage={linkage} dt={distance_threshold}.tiff',
		save_to_file=True,
	)
	_generate_many_dendrograms(
		data=cerealCN.get_frame(),
		linkage='complete',
		thresholds=complete_distance_thresholds,
		
		show=False,
		folder=_Q3D.folder_dendrograms,
		file_str_fmt= 'linkage={linkage} dt={distance_threshold}.tiff',
		save_to_file=True,
	)
	logger.info('best split is 0.80')
	# labeled = CleanNormalLabeled(cleanCereal=cerealCN, distance_threshold=0.8)
	# labeled.save()

	return