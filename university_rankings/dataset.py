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
from defs import QuestionTwoData as _Q2D

from helpers.dataset import (
	DatasetBase as _DatasetBase,
	DatasetCSV as _DatasetCSV,
	DatasetCSVReadOnly as _DatasetCSVReadOnly,
	DatasetSaveMixin as _DatasetSaveMixin,
)
import logging
logger = logging.getLogger(_Q2D.logger_name)
_Q2D.folder_dendrograms.mkdir(mode=0o775, parents=True,exist_ok=True)
_Q2D.folder_boxplots.mkdir(mode=0o775, parents=True, exist_ok=True)
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
			self.frame = _pd.DataFrame(normalizer.fit_transform(frame), columns=frame.columns)
		else:
			raise Exception('invalid path, rankings = None')

class CleanNormalLabeled(_DatasetBase, _DatasetSaveMixin):
	COLUMN_LABEL:_ClassVar[str] = 'LABEL'
	default_Path:_ClassVar[_Path] = _Q2D.CleanNormalLabeled
	aggcluster_default_args:_ClassVar[_Dict] = {
		'metric':'euclidean',
		'linkage':'complete',
		'compute_full_tree':True,
		'compute_distances':True,
		'n_clusters':None, # must be none when distance_threshold is set.
	}
	distance_threshold:float
	model:_AgglomerativeClustering

	def __init__(self, rankings:CleanNormalUniversity, distance_threshold:float):
		logger.debug('CleanNormalLabeled.__init__')
		super().__init__(path=self.default_Path, frame=None)
		self.distance_threshold = distance_threshold
		
		self.figure_file = _Q2D.folder_dendrograms.joinpath(f'dendrogram dt={self.distance_threshold:0.2f}.tiff')
		
		self.model = _AgglomerativeClustering(
			distance_threshold=self.distance_threshold,
			**self.aggcluster_default_args
		)
		self.frame = rankings.get_frame().copy()
		self.model.fit(self.frame)
		self.get_frame()[self.COLUMN_LABEL]=self.model.labels_
	
	def plotsave_dendrogram(self, save_to_file:bool=True, show:bool=False)->_Tuple[_Figure,_Axes]:
		logger.debug(f'plot_and_save_dendrogram({self.figure_file},...,...)')
		return _plot_dendrogram(self.figure_file, self.distance_threshold, self.model, save_to_file, show)
	
	def get_summary_statistics(self, save_to_file:bool=True, show:bool=False)->_Tuple[_pd.DataFrame, _pd.DataFrame]:
		''' Calculates mean & median with groupby=LABEL, saves and returns frames. '''
		frame_by_label = self.get_frame().groupby(by=self.COLUMN_LABEL)
		median = frame_by_label.median()
		mean = frame_by_label.mean()

		meanfig,ax = _plt.subplots()
		mean.plot(kind='bar',ax=ax)

		medianfig,ax=_plt.subplots()
		median.plot(kind='bar',ax=ax)
		if save_to_file:
			median.to_csv(_Q2D.CleanNormalMedian)
			mean.to_csv(_Q2D.CleanNormalMean)
			meanfig.savefig('meanfig.tiff')
			medianfig.savefig('medianfig.tiff')
			logger.info('saved CleanNormalMedian & Mean')
		if show:
			medianfig.show()
			meanfig.show()
		return median,mean
	
	def  plot_boxplots(self,save_to_file:bool=True, show:bool=False)->None:
		''' Creates a boxplot for subsets of the frame by LABEL. '''
		logger.info('Plotting features boxplots...')
		frame=self.get_frame()
		for label in frame[self.COLUMN_LABEL].unique():
			file = _Q2D.folder_boxplots.joinpath(f'university_cn boxplot label={label}.tiff')
			if file.exists() and save_to_file:
				logger.warning(f'File already exists. Skipping. {file}')
			fig,ax = _plt.subplots()
			fig.suptitle(f'UniversityCleanNormal Agglomerative Label: {label}')
			fig.set_dpi(500)
			fig.set_size_inches((10,10))
			label_frame = frame.loc[frame[self.COLUMN_LABEL]==label].drop(columns=[self.COLUMN_LABEL])
			label_frame.boxplot(
				ax=ax, rot=25
			)
			ax.set_ylim(0,1)
			if save_to_file:
				fig.savefig(file)
			if show:
				fig.show()

def generate_many_dendrograms(cleanNormalRankings:CleanNormalUniversity, save_to_file:bool, show:bool)->None:
	logger.info('CleanNormalUniversity.generate_many_dendrograms')
	CleanNormalLabeled(cleanNormalRankings, 0).plotsave_dendrogram(save_to_file=save_to_file,show=show)
	for it in [x/100 for x in range(40, 105, 5)]:
		if it==0:# skip
			continue
		logger.debug(f'distance_threshold:{it}')
		data = CleanNormalLabeled(cleanNormalRankings,it)
		if data.figure_file.exists() and save_to_file:
			logger.info(f'{data.figure_file} already exists. skipping...')
			continue
		data.plotsave_dendrogram(save_to_file=save_to_file,show=show)


def question_two():
	logger.info('===== Question Two =====')
	rankings = UniversityRankings.create_from_file()
	cleanNormal = CleanNormalUniversity(rankings=rankings)
	try:
		cleanNormal.save(clobber=False)
	except:
		logger.debug('cleanNormal data already exists, did not overwrite.')
		pass
	generate_many_dendrograms(cleanNormal, save_to_file=True, show=False)
	
	logger.info('A distance threshold of 0.80 was decided upon for our ideal model.')
	labeledData:CleanNormalLabeled = CleanNormalLabeled(cleanNormal, 0.80)
	labeledData.save()

	labeledData.get_summary_statistics(save_to_file=True)
	labeledData.plot_boxplots(save_to_file=True,show=False)
	logger.info('Quest 2.D information.')
	frame = rankings.get_frame().copy().loc[rankings.get_frame().index.isin(cleanNormal.get_frame().index)]
	frame[CleanNormalLabeled.COLUMN_LABEL] = labeledData.get_frame()[CleanNormalLabeled.COLUMN_LABEL]
	label_by_private_public = frame.groupby(by=CleanNormalLabeled.COLUMN_LABEL).value_counts(subset=[UniversityRankings.Columns.PublicPrivate])
	label_by_private_public.to_csv(_Q2D.PrivatePublicLabels,index=True)

	label_by_state = frame.groupby(by=CleanNormalLabeled.COLUMN_LABEL).value_counts(subset=[UniversityRankings.Columns.State])
	label_by_state.to_csv(_Q2D.StateLabels,index=True)

	logger.info('========================')
	return