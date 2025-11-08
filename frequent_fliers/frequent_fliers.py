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
from defs import QuestionFourData as _Q4D

from helpers.dataset import (
	DatasetBase as _DatasetBase,
	DatasetCSV as _DatasetCSV,
	DatasetCSVReadOnly as _DatasetCSVReadOnly,
	DatasetSaveMixin as _DatasetSaveMixin,
)
import logging
logger = logging.getLogger(_Q4D.logger_name)
_Q4D.folder_figures.mkdir(mode=0o775, parents=True,exist_ok=True)
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
	format_string = 'dendrogram metric={metric} linkage={linkage} dt={distance_threshold:05.2f}.tiff'
	thresholds:_List[float] = [ 
		3.0, 3.2,
		3.4, 3.6, 4.0,
		5.0, 5.4, 5.5,
		5.6, 5.8, 7.0,
		12.0,14.0,
	]
	
	_generate_many_dendrograms(
		data=ffn.get_frame(), thresholds=thresholds,
		folder=_Q4D.folder_figures, file_str_fmt=format_string,
		linkage='ward', metric='euclidean',
		save_to_file=True, raise_err_exists=False, clobber=False
	)
	# ffl = _FrequentFliersLabeled(ffn, dendrogram=_CustomDendrogram(
	# 	file=_Path('tmp.csv'), distance_threshold=0,linkage='complete',
	# 	save_to_file=True, raise_err_exists=False, clobber=True, metric='ward',
	# 	data=ffn.get_frame()
	# ))
	# ffl.dendrogram.file = _Q4D.folder_figures.joinpath(
	# 	ffl.dendrogram.format_from_vars(format_string)
	# )