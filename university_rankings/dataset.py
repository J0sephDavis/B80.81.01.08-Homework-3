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

aggcluster_default_args:_Dict = {
	'metric':'euclidean',
	'linkage':'complete',
	'compute_full_tree':True,
	'compute_distances':True,
	'n_clusters':None, # must be none when distance_threshold is set.
}

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
			self.frame = _pd.DataFrame(normalizer.fit_transform(frame), columns=frame.columns)
		else:
			raise Exception('invalid path, rankings = None')
		
	def plot_and_save_dendrogram(self,
				file:_Path,
				model:_AgglomerativeClustering,
				show:bool=False,
				dendrogram_args:_Dict={}
			)->_Tuple[_Figure, _Axes]:
		# Plotting code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
		logger.debug(f'plot_and_save_dendrogram({file},...,...)')
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
		logger.debug('call dendrogram plotter')	
		_dendrogram(
			Z= _np.column_stack([model.children_,model.distances_, counts]),
			ax=ax,
			**dendrogram_args
		)
		
		logger.info(f'saving dendrogram to {file}.')
		fig.savefig(fname=str(file))
		if show:
			fig.show()
		return fig,ax

	def generate_many_dendrograms(self)->None:
		logger.info('CleanNormalUniversity.generate_many_dendrograms')
		dendrogram_folder=_Q2D.DendrogramFolder
		dendrogram_folder.mkdir(mode=0o775, parents=True,exist_ok=True)
		self.plot_and_save_dendrogram(
			file=_Path('dendrogram dt=0.tiff'),
			model= _AgglomerativeClustering(
				distance_threshold=0,
				**aggcluster_default_args
			)
		)
		for it in [x/100 for x in range(40, 105, 5)]:
			if it==0:# skip
				continue
			logger.debug(f'distance_threshold:{it}')
			file=dendrogram_folder.joinpath(f'dendrogram dt={it:0.2f}.tiff')
			if file.exists():
				logger.info(f'{file} already exists. skipping...')
				continue
			self.plot_and_save_dendrogram(
				file=file,
				model= _AgglomerativeClustering(
					distance_threshold=it,
					**aggcluster_default_args
				),
				dendrogram_args= {
					# 'truncate_mode':'lastp',
					'color_threshold':it
				}
			)
	

def question_two():
	logger.info('===== Question Two =====')
	rankings = UniversityRankings.create_from_file()
	cleanNormal = CleanNormalUniversity(rankings=rankings)
	try:
		cleanNormal.save(clobber=False)
	except:
		logger.debug('cleanNormal data already exists, did not overwrite.')
		pass
	cleanNormal.generate_many_dendrograms() # Generate our data for viewing
	logger.info('create ideal model')
	
	ideal_model = _AgglomerativeClustering(distance_threshold=0.80, **aggcluster_default_args)
	frame = cleanNormal.get_frame().copy()
	frame['LABELS'] = ideal_model.fit_predict(frame)

	frame_by_label = frame.groupby(by='LABELS')

	median_stats = frame_by_label.median()
	mean_stats = frame_by_label.mean()

	logger.info(f'frame_by_label:{frame_by_label}')
	logger.info(f'mean_stats:{mean_stats}')
	logger.info(f'median_stats:{median_stats}')
	logger.info('A distance threshold of 0.80 was decided upon for our ideal model.')
	# compare_clusters(n, cleanNormal)
	logger.info('========================')
	return