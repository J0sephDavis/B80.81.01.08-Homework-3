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
)
import pandas as _pd
from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram as _dendrogram
import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes
import matplotlib.patches as _patches

def plot_dendrogram(
			file:_Path, distance_threshold:float,
			model:_AgglomerativeClustering,
			save_to_file:bool, show:bool
		)->_Tuple[_Figure,_Axes]:
		''' Generate a dendrogram from the pre-fit model. '''
		logger.debug(f'plot_and_save_dendrogram({file},...,...)')
		# Plotting code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
		# Generate linkage table
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
		ax.set_title(f'distance_threshold={distance_threshold:0.2f}')
		logger.debug('call dendrogram plotter')	
		dend = _dendrogram(
			Z= _np.column_stack([model.children_,model.distances_, counts]),
			ax=ax,
			color_threshold=distance_threshold
		)
		# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
		leafa = dend['leaves']
		ordered_clust = model.labels_[leafa]
		leaf_colors=dend['leaves_color_list']
		color_map = {}
		for lbl,color in zip(ordered_clust, leaf_colors):
			color_map.setdefault(lbl,color)
		if len(color_map) < 50:
			# If there are too many clusters you can override this, but its ugly.
			legend_patches = [
				_patches.Patch(color=color_map[k],label=f'C{k}')
				for k in sorted(color_map)
			]
			ax.legend(
				handles= legend_patches,
				title=f'Clusters ({len(color_map)})',
				loc='upper center',
				frameon=True,
				bbox_to_anchor=(0.5,1.05),
				ncol=8,
				fancybox=True,
				shadow=True
			)

		if save_to_file:
			fig.savefig(fname=str(file))
		if show:
			fig.show()
		return fig,ax