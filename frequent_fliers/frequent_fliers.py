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
from sklearn.neighbors import (
	NearestNeighbors as _NearestNeighbors
)
from sklearn.cluster import (
	KMeans as _KMeans,
	DBSCAN as _DBSCAN
)
import numpy as _np
from defs import QuestionFiveData as _Q5D

from helpers.dataset import (
	DatasetBase as _DatasetBase,
	DatasetCSV as _DatasetCSV,
	DatasetCSVReadOnly as _DatasetCSVReadOnly,
	DatasetSaveMixin as _DatasetSaveMixin,
)
import logging
logger = logging.getLogger(_Q5D.logger_name)
_Q5D.folder_figures.mkdir(mode=0o775, parents=True,exist_ok=True)
_Q5D.folder_datasets.mkdir(mode=0o775, parents=True,exist_ok=True)
import matplotlib.pyplot as _plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes

from .dataset import (
	FrequentFliers as _FrequentFliers,
	FrequentFliersNormalized as _FrequentFliersNormalized,
	FrequentFliersLabeled as _FrequentFliersLabeled,
)

def question_four():
	ff = _FrequentFliers()
	ffn = _FrequentFliersNormalized(ff)
	try: ffn.save(clobber=False)
	except FileExistsError: pass
	format_string = 'metric={metric} linkage={linkage} dt={distance_threshold:05.2f}.tiff'
	thresholds:_List[float] = [ 
		3.0, 3.2,
		3.4, 3.6, 4.0,
		5.0, 5.4, 5.5,
		5.6, 5.8, 7.0,
		12.0,14.0,
	]
	
	_generate_many_dendrograms(
		data=ffn.get_frame(), thresholds=thresholds,
		folder=_Q4D.folder_figures, file_str_fmt=f'dendrogram {format_string}',
		linkage='ward', metric='euclidean',
		save_to_file=True, raise_err_exists=False, clobber=False
	)
	def generate_dendrogram(sample_frac:float):
		sample = _FrequentFliersNormalized(
			frequent_fliers=None,
			frame=ffn.get_frame().sample(frac=sample_frac)
		)
		ffl = _FrequentFliersLabeled(
			ffNorm=sample,
			file_path=_Q4D.folder_datasets.joinpath(f'Labeled sample={sample_frac:04.2f}.csv'),
			dendrogram=_CustomDendrogram(
				file=_Path('tmp.csv'), distance_threshold=5.8,linkage='ward',
				save_to_file=True, raise_err_exists=False, clobber=True, metric='euclidean',
				data=sample.get_frame()
			)
		)
		ffl.dendrogram.file = _Q4D.folder_figures.joinpath(
			ffl.dendrogram.format_from_vars(f'dendrogram sample_frac={sample_frac} {format_string}')
		)
		median,mean = ffl.get_summary_statistics()
		mean.to_csv(
			_Q4D.folder_datasets.joinpath(
				ffl.dendrogram.format_from_vars(
					f'Labeled Mean (sample:{sample_frac}) metric={{metric}} linkage={{linkage}} dt={{distance_threshold:05.2f}}.csv'
				)
			)
		)
		return ffl
	ffl_full_sample = generate_dendrogram(sample_frac=1.0)
	ffl_full_sample.save()
	ffl_95_sample = generate_dendrogram(sample_frac=0.95)
	ffl_95_sample.save()

	frame = ffl_full_sample.get_frame().drop(columns=_FrequentFliersLabeled.COLUMN_LABEL)
	kmeans = _KMeans(n_clusters=5).fit(frame)
	frame[_FrequentFliersLabeled.COLUMN_LABEL] = kmeans.labels_
	frame_by_label = frame.groupby(by=_FrequentFliersLabeled.COLUMN_LABEL)
	mean = frame_by_label.mean()
	frame.to_csv(_Q4D.folder_datasets.joinpath('kmeans_labeled.csv'))
	mean.to_csv(_Q4D.folder_datasets.joinpath('kmeans_mean.csv'))

	comparison_frame = frame.rename(
		columns={
			_FrequentFliersLabeled.COLUMN_LABEL:'KMEANS_LABEL',
		}
	)
	comparison_frame['HC_LABEL']=ffl_full_sample.get_frame()[_FrequentFliersLabeled.COLUMN_LABEL]
	comparison_frame = comparison_frame[['HC_LABEL','KMEANS_LABEL']]
	comparison_frame.to_csv(
		_Q4D.folder_datasets.joinpath('label_comparison_kmeans_hc_full.csv')
	)

	comparison_frame.groupby(by='HC_LABEL').value_counts().to_csv(
		_Q4D.folder_datasets.joinpath('label_comparison_kmeans_hc_full_valcount.csv')
	)

	return

def create_elbow(frame):
	n_neighbors=5
	neighbors = _NearestNeighbors(n_neighbors=n_neighbors)
	estimator = neighbors.fit(frame)
	distances,index=estimator.kneighbors(frame,n_neighbors)
	kth_distances = _np.sort(distances[:,n_neighbors-1])
	fig,ax = _plt.subplots()
	ax.plot(kth_distances)
	ax.set_ylabel('Distance')
	ax.set_xlabel('Points')
	ax.set_title('k=5 k-Distance graph')
	ax.axhline(0.020)
	ax.set_ybound(0,0.1)
	ax.set_xbound(lower=0)
	ax.grid(visible=True, which='both', axis='x')
	_plt.close(fig=fig)
	return fig,ax

def run_dbscan(frame, eps:float, file:_Path):
	logger.debug(f'run_dbscan, eps={eps} file={file}')
	dbscan  =_DBSCAN(eps=eps, min_samples=5)
	dbscan.fit(frame)
	return dbscan

def question_five():
	ff = _FrequentFliers()
	ffn = _FrequentFliersNormalized(ff)
	frame = ffn.get_frame()
	elbow_file = _Q5D.folder_figures.joinpath('kNN elbow.tiff')
	dbscan_labels_valcount = _Q5D.folder_datasets.joinpath('dbscan label val count.csv')
	dbscan_labels = _Q5D.folder_datasets.joinpath('dbscan_labeled.csv')
	dbscan_centroids = _Q5D.folder_datasets.joinpath('dbscan_centroids.csv')
	
	if not elbow_file.exists():
		logger.info('finding elbow')
		elbow_graph = create_elbow(frame=frame)
		elbow_graph[0].savefig(elbow_file)
	
	logger.info('Optimal epsilon is about 0.02?')
	eps =0.02
	if not (dbscan_labels_valcount.exists() and dbscan_labels.exists() and dbscan_centroids.exists()):
		logger.info('dbscan')
		dbscan = run_dbscan(frame,eps,_Q5D.folder_figures.joinpath(f'dbscan eps={eps}.tiff'))
		labeled_frame = (frame.copy())
		labeled_frame['DBSCAN']=dbscan.labels_
		valcnt = labeled_frame['DBSCAN'].value_counts()
		logger.info('saving dbscan data')
		valcnt.to_csv(dbscan_labels_valcount)
		labeled_frame.to_csv(dbscan_labels)
		labeled_frame.groupby(by='DBSCAN').mean().to_csv(dbscan_centroids)

	return