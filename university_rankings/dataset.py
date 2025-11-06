from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
from typing import (
	Optional as _Optional,
	List as _List,
	Union as _Union,
	ClassVar as _ClassVar,
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

class UniversityRankings(_DatasetCSVReadOnly):
	default_path:_ClassVar[_Path] = _Q2D.Rankings
	logger.debug(f'UniversityRankings.default_path: {default_path}')
	def __init__(self):
		logger.debug('UniversityRankings.__init__')
		super().__init__(path=self.default_path, frame=None)

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
			]).dropna(how='all', axis=0)
			normalizer = _Normalizer()
			self.frame = _pd.DataFrame(normalizer.fit_transform(frame))
		else:
			raise Exception('invalid path, rankings = None')
	
	def plot_dendrogram(self):
		''' perf hierarhical clustering
		- Linkage:Complete
		- Distance:eculidean
		logger.debug('CleanNormalUniversity.plot_dendrogram')
		hc_model = _AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=0,
			compute_full_tree=True,

			linkage='complete',
			metric='euclidean',
		)
		# Plotting code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
		hc_model.fit(self.get_frame())
		counts = _np.zeros(hc_model.children_.shape[0])
		n_samples = len(hc_model.labels_)
		for i, merge in enumerate(hc_model.children_):
			current_count = 0
			for child_idx in merge:
				if child_idx < n_samples:
					current_count += 1  # leaf node
				else:
					current_count += counts[child_idx - n_samples]
		counts[i] = current_count
		logger.debug(f'hc_model - counts:{counts}')
		logger.debug(f'hc_model - n_samples:{n_samples}')
		logger.debug(f'hc_model.children_:{hc_model.children_}')
		logger.debug(f'hc_model.distances_:{hc_model.distances_}')
		fig,ax = _plt.subplots(figsize=(10,10))
		_dendrogram(_np.column_stack([
			hc_model.children_,hc_model.distances_, counts
		]))

def question_two():
	rankings = UniversityRankings.create_from_file()
	cleanNormal = CleanNormalUniversity(rankings=rankings)
	cleanNormal.save()
	
	cleanNormal.plot_dendrogram()
	return