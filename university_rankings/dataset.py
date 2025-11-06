from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
from typing import (
	Optional as _Optional,
	List as _List,
	Union as _Union,
	ClassVar as _ClassVar,
	Tuple as _Tuple,
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
	
	def agglomerative_cluster(self,
				n_clusters:_Optional[int],
				distance_threshold:_Optional[int]
			)->_AgglomerativeClustering:
		''' Returns the model created through agglomerative modeling. 
		n_clusters -- the number of clusters
		'''
		if n_clusters is None == distance_threshold is None:
			raise Exception('Only one may be set')
		logger.debug(f'CleanNormalUniversity.agglomerative_cluster(k={n_clusters}, dt={distance_threshold})')
		if n_clusters is None:
			return _AgglomerativeClustering(
				n_clusters=None,
				distance_threshold=distance_threshold,
				metric='euclidean',
				linkage='complete',
				compute_full_tree=True,
				compute_distances=True,
			)
		elif distance_threshold is None:
			return _AgglomerativeClustering(
				n_clusters=n_clusters,
				metric='euclidean',
				linkage='complete',
				compute_full_tree=True,
				compute_distances=True,
			)
		raise Exception('unintended')

	def plot_dendrogram(self,
				model:_AgglomerativeClustering,
			)->_Tuple[_Figure, _Axes]:
		''' perf hierarhical clustering
		Arguments:
		model -- If you have already made the model, provide it. Ignore n_clusters & distance threshold arguments
		n_clusters -- given to AgglomerativeClustering
		distance_threshold -- given to AgglomerativeClustering
		
		- Linkage:Complete
		- Distance:eculidean
		- Normalized data
		Returns the dendrogram plot & model
		'''
		logger.debug('CleanNormalUniversity.plot_dendrogram')
		# Plotting code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
		model.fit(self.get_frame())
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
		_dendrogram(_np.column_stack([
			model.children_,model.distances_, counts
		]), ax=ax)
		return fig,ax

	
def plot_and_save_dendrogram(
			file:_Path,
			data:CleanNormalUniversity,
			model:_AgglomerativeClustering,
			show:bool=False
		)->None:
	logger.debug(f'plot_and_save_dendrogram({file},...,...)')
	fig,ax = data.plot_dendrogram(model)
	logger.info(f'saving dendrogram to {file}.')
	fig.savefig(fname=str(file))
	if show:
		fig.show()
	return

def question_two():
	logger.info('===== Question Two =====')
	rankings = UniversityRankings.create_from_file()
	cleanNormal = CleanNormalUniversity(rankings=rankings)
	cleanNormal.save()
	
	d_fig, d_ax, model = cleanNormal.plot_dendrogram()
	# d_fig.show()
	logger.info('saving dendrogram.')
	d_fig.savefig(fname=_Q2D.DendrogramFigure)
	logger.info(f'clusters: {model.n_clusters_}')
	logger.warning('we must somehow estimate the proper amount of clusters....')
	
	n_clusters = [2,4,6]
	for n in n_clusters:
		compare_clusters(n, cleanNormal)
	logger.info('========================')
	return