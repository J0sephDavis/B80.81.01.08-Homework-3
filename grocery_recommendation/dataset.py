from typing import (
	Optional as _Optional,
	List as _List,
)
import pandas as _pd
from helpers.dataset import DatasetBase as _DatasetBase
from helpers.exceptions import DatasetInvalidPath as _DatasetInvalidPath

from defs import (
	FILEPATH_RECOMMENDATION_DATA as _FILEPATH_RECOMMENDATION_DATA,
	FILEPATH_GROCERIES as _FILEPATH_GROCERIES
)

from mlxtend.frequent_patterns import apriori as _apriori
from mlxtend.frequent_patterns import association_rules as _association_rules

class GroceriesDataset(_DatasetBase):
	''' Class representing the original groceries dataset'''
	def __init__(self)->None:
		super().__init__(_FILEPATH_GROCERIES)

	def load(self, force_reload: bool = False, **read_csv_kwargs) -> None:
		''' We override the origina load method because this data is not rectangular. '''
		if (not force_reload) and (self.frame is not None):
			return
		if self.path is None:
			raise _DatasetInvalidPath('Cannot load frame, Path is None.')
		if not self.path.exists():
			raise FileNotFoundError('Cannot load frame, Dataset file not found.')
		# ----
		with open(self.path, 'r') as file:
			records = [(line.strip(),) for  line in file]
		self.frame = _pd.DataFrame.from_records(data=records, columns=['Transaction'])
		return
	
	def save(self, clobber:bool=True, include_index:bool=False, **to_csv_kwargs)->None:
		''' You cannot overwrite the original dataset. '''
		raise Exception('Forbidden')
	
	def hot_encode_transactions(self)->_pd.DataFrame: # QUESTION 1.A
		''' Used to hotencode transactions when making the recommendation dataset. '''
		frame = self.get_frame()
		dumm = frame['Transaction'].str.get_dummies(sep=',').astype(bool)
		return dumm

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
	
	def get_frequent_itemsets(self)->_pd.DataFrame: # QUESTION 1.B - Generate Frequent Itemsets
		return _apriori(self.get_frame(), min_support=0.01, use_colnames=True)

	
from enum import (
	StrEnum as _StrEnum,
	auto as _auto
)

class AssRules():
	''' Association Rules''' # Question 1.C - Generate Association Rules
	rules:_pd.DataFrame # The association rules

	def __init__(self, frequent_itemset:_pd.DataFrame):
		self.rules =  _association_rules(
			frequent_itemset,
			metric='confidence',
			min_threshold=0.2,
		)

	class cols(_StrEnum):
		''' Columns we want to print'''
		antecedents=_auto()
		consequents=_auto()
		support=_auto()
		confidence=_auto()
		lift=_auto()
	
	def print_rules(self):
		''' Print the association rules '''
		msg:str = 'Question 1.C - Association Rules'
		sep:str = len(msg)*'-'
		print(msg,sep,sep='\n')
		for rec in self.rules[[m.value for m in self.cols]].itertuples(index=False):
			output:_List[str] = []
			for col in self.cols:
				output.append(f'{col.value}: {getattr(rec,col.value)}')
			print(' | '.join(output))
		print(sep)
	
	def get_rules(self):
		return self.rules

def recommend_items(transaction:_List[str])->_List[str]:

	return
def question_one():
	groceries = GroceriesDataset()
	recset = RecommendationDataset.from_grocery_data(groceries) # For Question One.
	
	# Question 1.B. ---
	freqset = recset.get_frequent_itemsets()
	top_ten = freqset.sort_values(by='support',ascending=False).iloc[0:10]
	msg:str = 'Question 1.B - Top Ten Frequent Itemsets by Support:'
	sep:str = len(msg)*'-'
	print(msg,sep,sep='\n')
	for rec in top_ten.itertuples(index=False):
		print(f'Support: {rec[0]:.3f}\t{rec[1]}')
	print(sep)
	
	# Question 1.C. ----
	ar = AssRules(freqset)
	ar.print_rules()
	return