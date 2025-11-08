import logging
from defs import app_logger_name
logger = logging.getLogger(f'{app_logger_name}.plotting')
from typing import (
	Tuple as _Tuple,
)

from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
from typing import (
	Optional as _Optional,
	List as _List,
	Union as _Union,
	ClassVar as _ClassVar,
	Tuple as _Tuple,
	Dict as _Dict,
	Literal as _Literal,
)
import pandas as _pd
from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram as _dendrogram
import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes
import matplotlib.patches as _patches
from dataclasses import (
	dataclass as _dataclass,
	field as _field
)

def create_linkage_matrix(model:_AgglomerativeClustering):
	''' From https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html '''
	logger.debug(f'create_linkage_matrix')
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
	return _np.column_stack(
		[model.children_, model.distances_, counts]
	).astype(float)

@_dataclass
class CustomDendrogram():
	''' can be used like a monad '''
	model:_Optional[_AgglomerativeClustering] = _field(default=None, init=False)
	file:_Path # Where the figure will be saved
	distance_threshold:float # used for clustering & plotting
	linkage:_Literal['single','complete']
	data:_pd.DataFrame # The dataframe we want to fit our clusters to.
	# User may care.
	save_to_file:bool = _field(default=True) # Should we save to file on a call to plot_dendrogram?
	clobber:bool = _field(default=False)
	raise_err_exists:bool = _field(default=True) # raise err if file exists and clobber=False
	show:bool = _field(default=False) # Should we show the figure or close the figure during plot_dendrogram?

	# default values that probably do not change
	metric:str = _field(default='euclidean')
	compute_full_tree:bool = _field(default=True)
	compute_distances:bool = _field(default=True)
	n_clusters:_Optional[int] = _field(default=None)

	figure_title:str = _field(init=False)
	figure_size:_Tuple[float,float] = _field(default=(10,10))
	def __post_init__(self):
		logger.debug(f'{self.__class__}.__post_init__')
		self.figure_title = f'linkage={self.linkage} distance_threshold={self.distance_threshold:0.2f}'

	def get_model(self, force_refresh:bool=False):
		logger.debug(f"{self.__class__}.create_model\n{self}")
		if not force_refresh and self.model is not None:
			return self.model
		logger.info('generating model')
		self.model = _AgglomerativeClustering(
			linkage=self.linkage,
			distance_threshold=self.distance_threshold,
			metric=self.metric,
			compute_full_tree=self.compute_full_tree,
			compute_distances=self.compute_distances,
			n_clusters=self.n_clusters,
		)
		self.model.fit(self.data)
		return self.model
	
	def format_from_vars(self, fmt_str:str):
		return fmt_str.format(**vars(self))
	
	def _create_legend(self, dendrogram_data:_Dict, fig:_Figure, ax:_Axes):
		''' Creates the legend based on the cluster labels in the model,
		 and the colors that scipy used.
		'''
		logger.debug(f"{self.__class__}._create_legend")
		# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
		leafa = dendrogram_data['leaves']
		ordered_clust = self.get_model().labels_[leafa]
		leaf_colors=dendrogram_data['leaves_color_list']
		color_map = {}
		for lbl,color in zip(ordered_clust, leaf_colors):
			color_map.setdefault(lbl,color)
		if len(color_map) < 50:
			# If there are too many clusters you can override this, but its ugly.
			legend_patches = [
				_patches.Patch(color=color_map[k],label=f'C{k}')
				for k in sorted(color_map)
			]
			fig.legend(
				handles= legend_patches,
				title=f'Clusters ({len(color_map)})',
				loc='upper center',
				frameon=True,
				ncol=8,
				fancybox=True,
				shadow=True
			)
		else:
			logger.debug('color map >= 50, not drawing legend.')

	def plot_dendrogram(self)->_Optional[_Tuple[_Figure,_Axes]]:
		''' Plots the dendrogram data'''
		if self.save_to_file and not self.show:
			if not self.clobber and not self.raise_err_exists and self.file.exists():
				logger.info(f'plot_dendrogram. skipping... File ({self.file}) exists, clobber=False, raise_err=False, show=False')
				return None
		logger.debug(f"{self.__class__}.plot_dendrogram")
		fig,ax=_plt.subplots(figsize=self.figure_size)
		ax.set_title(self.figure_title)
		
		dend = _dendrogram(
			Z=create_linkage_matrix(self.get_model()),
			ax=ax,
			color_threshold=self.distance_threshold
		)
		self._create_legend(dend, fig, ax)
		if self.save_to_file:
			if self.file.exists():
				if self.clobber:
					fig.savefig(fname=str(self.file))
				elif self.raise_err_exists:
					raise FileExistsError(f'Cannot create dendrogram, file already exists. {self.file}')
				else:
					logger.warning(f'File already exists, but not raising err: {self.file}')
			else:
				fig.savefig(fname=str(self.file))
		
		fig.show() if self.show else _plt.close(fig=fig)
		return fig,ax
	

def generate_many_dendrograms(
		data:_pd.DataFrame,
		thresholds:_List[float],
		folder:_Optional[_Path] = None,
		file_str_fmt:str = 'linkage={linkage} dt={distance_threshold:0.2f}',
		**init_kwargs
		)->_List[CustomDendrogram]:
	def make_file_name(linkage:str,distance_threshold:float)->_Path:
		# TODO: just pass kwargs/dict as an argument so the fmt string could be anything related to our obj.
		basename:str = file_str_fmt.format(linkage=linkage,distance_threshold=distance_threshold)
		if folder is not None:
			return folder.joinpath(basename)
		return _Path(basename)
	plots = [
		CustomDendrogram(
			distance_threshold=thresh,
			data=data,
			file = make_file_name(
				linkage=init_kwargs.get('linkage'),  distance_threshold=thresh
			),
			**init_kwargs
		)
		for thresh in thresholds
	]
	for plot in plots:
		plot.plot_dendrogram()
	return plots