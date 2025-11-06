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

	def save(self, clobber:bool=True, include_index:bool=False, **to_csv_kwargs)->None:
		''' You cannot overwrite the original dataset. '''
		raise Exception('Forbidden')
	
	def get_clean_copy(self)->_pd.DataFrame:
		''' returns a copy of the dataset without any null values (removed entire record)'''
		return self.get_frame().copy().dropna(axis=0, how='any')