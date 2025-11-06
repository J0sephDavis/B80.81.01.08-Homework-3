from typing import (
	Optional as _Optional,
	List as _List,
	Tuple as _Tuple,
	Union as _Union,
)
from enum import StrEnum as _StrEnum
from pathlib import Path as _Path
import pandas as _pd
from helpers.dataset import DatasetBase as _DatasetBase
from groceries import GroceriesDataset
from defs import FILEPATH_RECOMMENDATION_DATA as _FILEPATH_RECOMMENDATION_DATA
from mlxtend.frequent_patterns import apriori as _apriori
class RecommendationDataset(_DatasetBase):
	''' Class representing the recommendation data'''
	def __init__(self,frame:_Optional[_pd.DataFrame] = None)->None:
		''' Initializes the dataset with a default file path and the frame, if given.'''
		super().__init__(
			path=_FILEPATH_RECOMMENDATION_DATA,
			frame=frame,
			nrows=None,
			use_columns=None,
		)
	@staticmethod
	def from_grocery_data(groceries:GroceriesDataset):
		''' Hot encodes the grocery data. '''
		return RecommendationDataset(
			frame= groceries.hot_encode_transactions()
		)
	
	def get_frequent_itemsets_apriori(self)->_pd.DataFrame: # QUESTION 1.B - Generate Frequent Itemsets
		freqset = _apriori(self.get_frame(), min_support=0.01)
		# FOR ASSIGNMENT
		freqset_COLNAMES = _apriori(self.get_frame(), min_support=0.01, use_colnames=True)
		top_ten = freqset_COLNAMES.sort_values(by='support',ascending=False).iloc[0:10]
		msg:str = 'Question 1.B - Top Ten Frequent Itemsets by Support:'
		sep:str = len(msg)*'-'
		print(msg,sep,sep='\n')
		for rec in top_ten.itertuples(index=False):
			print(f'Support: {rec[0]:.3f}\t{rec[1]}')
		print(sep)
		return freqset