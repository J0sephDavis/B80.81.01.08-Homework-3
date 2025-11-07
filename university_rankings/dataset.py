from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
from typing import (
	Optional as _Optional,
	List as _List,
	Union as _Union,
	ClassVar as _ClassVar,
	Tuple as _Tuple,
	Dict as _Dict,
)
import pandas as _pd
from helpers.dataset import DatasetBase as _DatasetBase
from helpers.exceptions import DatasetInvalidPath as _DatasetInvalidPath
from sklearn.preprocessing import Normalizer as _Normalizer
from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram as _dendrogram
import numpy as _np
from defs import QuestionTwoData as _Q2D

from helpers.dataset import (
	DatasetBase as _DatasetBase,
	DatasetCSV as _DatasetCSV,
	DatasetCSVReadOnly as _DatasetCSVReadOnly,
	DatasetSaveMixin as _DSaveMixin,
	DatasetLoadCSVMixin as _DLoadCSVMixin,
	DatasetTextFileMixin as _DatasetTextFileMixin,
)
import logging
logger = logging.getLogger(_Q2D.logger_name)
import matplotlib.pyplot as _plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes

class UniversityRankings(_DatasetCSVReadOnly):
	default_path:_ClassVar[_Path] = _Q2D.Rankings
	logger.debug(f'UniversityRankings.default_path: {default_path}')
	def __init__(self, path:_Path=default_path):
		logger.debug('UniversityRankings.__init__')
		super().__init__(path=path, frame=None)

	@classmethod
	def create_from_file(cls,path=default_path)->'UniversityRankings':
		logger.debug(f'UniversityRankings.create_from_file(path={path})')
		return super().create_from_file(path)
	
	class Columns(_StrEnum):
		CollegeName = r'College Name'
		State = r'State'
		PublicPrivate = r'Public (1)/ Private (2)'
		AppRecieved = r'# appli. rec\'d'
		AppAccept = r'# appl. accepted'
		NewStudentsEnrolled = r'# new stud. enrolled'
		NewStudentsFromTop10P = r'% new stud. from top 10%'
		NewStudentsFromTop25P = r'% new stud. from top 25%'
		FTUndergrad = r'# FT undergrad'
		PTUndergrad = r'# PT undergrad'
		InStateTuition = r'in-state tuition'
		OutOfStateTuition = r'out-of-state tuition'
		Room = r'room'
		Booard = r'board'
		AdditionalFees = r'add. fees'
		EstimatedBookCosts = r'estim. book costs'
		EstimatedPersonalCost = r'estim. personal $'
		RatioFacultyWithPHD = r'% fac. w/PHD'
		RatioStudentFaculty = r'stud./fac. ratio'
		GraduationRate = r'Graduation rate'

class CleanNormalUniversity(_DatasetCSV):
	default_path:_ClassVar[_Path] = _Q2D.CleanNormal
	logger.debug(f'CleanNormalUniversity.default_path: {default_path}')
	def __init__(self, rankings:_Optional[UniversityRankings]=None):
		logger.debug(f'CleanNormalUniversity.__init__(rankings={rankings})')
		super().__init__(path=self.default_path, frame=None)
		if self.path is not None and self.path.exists():
			logger.debug('CleanNormalUniversity.__init__ - loaded from file')
			self.load()
		elif rankings is not None:
			logger.debug('CleanNormalUniversity.__init__ - cleaning & normalizing from dataset')
			frame:_pd.DataFrame = rankings.get_frame().drop(columns=[
				UniversityRankings.Columns.CollegeName,
				UniversityRankings.Columns.PublicPrivate,
				UniversityRankings.Columns.State,
			]).dropna(how='any', axis=0)
			normalizer = _Normalizer()
			self.frame = _pd.DataFrame(normalizer.fit_transform(frame))
		else:
			raise Exception('invalid path, rankings = None')


# def compare_clusters(data:CleanNormalUniversity, n_clusters:_Optional[int]=None, distance_threshold:int = 0):
# 	logger.info(f'compare_clusters(n_clusters={n_clusters})')
# 	model = data.agglomerative_cluster(n_clusters, distance_threshold)
# 	frame = data.get_frame().copy()
# 	labels = model.fit_predict(frame)
# 	frame['LABELS']=labels
# 	frame_by_label = frame.groupby(by='LABELS')
# 	median_stats = frame_by_label.median()
# 	mean_stats = frame_by_label.mean()
# 	logger.info(f'frame_by_label:{frame_by_label}')
# 	logger.info(f'mean_stats:{mean_stats}')
# 	logger.info(f'median_stats:{median_stats}')
# 	return
	
def plot_and_save_dendrogram(
			file:_Path,
			data:CleanNormalUniversity,
			model:_AgglomerativeClustering,
			show:bool=False
		):
	# Plotting code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
	logger.debug(f'plot_and_save_dendrogram({file},...,...)')

	model.fit(data.get_frame())
	counts = _np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
	for i, merge in enumerate(model.children_):
		current_count = 0
		for child_idx in merge:
			if child_idx < n_samples:
				current_count += 1  # leaf node
			else:
				current_count += counts[child_idx - n_samples]
	counts[i] = current_count
	fig,ax = _plt.subplots(figsize=(10,10))

	logger.debug('call dendrogram plotter')	
	_dendrogram(_np.column_stack([
		model.children_,model.distances_, counts
	]), ax=ax)
	
	logger.info(f'saving dendrogram to {file}.')
	fig.savefig(fname=str(file))
	if show:
		fig.show()
	return fig,ax

aggcluster_default_args:_Dict = {
	'metric':'euclidean',
	'linkage':'complete',
	'compute_full_tree':True,
	'compute_distances':True,
}

def question_two():
	logger.info('===== Question Two =====')
	rankings = UniversityRankings.create_from_file()
	cleanNormal = CleanNormalUniversity(rankings=rankings)
	cleanNormal.save()
	
	plot_and_save_dendrogram(
		file=_Path('dendrogram k=None dt=0.tiff'),
		data= cleanNormal,
		model= _AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=0,
			**aggcluster_default_args
		)
	)
	ideal_threshold:float = 0.2
	logger.info('We have identified the distance threshold, by observation, as 0.2')
	for it in [x/10 for x in range(0, 10, 1)]:
		logger.debug(f'x:{it}')
		plot_and_save_dendrogram(
			file=_Path(f'dendrogram dt={it}.tiff'),
			data= cleanNormal,
			model= _AgglomerativeClustering(
				n_clusters=None,
				distance_threshold=it,
				**aggcluster_default_args
			)
		)
	# compare_clusters(n, cleanNormal)
	logger.info('========================')
	return