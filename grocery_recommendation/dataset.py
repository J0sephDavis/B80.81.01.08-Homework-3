from typing import (
	Optional as _Optional,
	List as _List,
	ClassVar as _ClassVar,
)
import pandas as _pd
from helpers.dataset import (
	DatasetBase as _DatasetBase,
	DatasetCSV as _DatasetCSV,
	DatasetCSVReadOnly as _DatasetCSVReadOnly,
	DatasetSaveMixin as _DSaveMixin,
	DatasetLoadCSVMixin as _DLoadCSVMixin,
	DatasetTextFileMixin as _DatasetTextFileMixin,
)
from helpers.exceptions import DatasetInvalidPath as _DatasetInvalidPath
from pathlib import Path as _Path
from defs import (
	QuestioneOneData as _Q1D
)

from mlxtend.frequent_patterns import apriori as _apriori
from mlxtend.frequent_patterns import association_rules as _association_rules
from enum import (
	StrEnum as _StrEnum,
	auto as _auto
)


class GroceriesDataset(_DatasetBase, _DatasetTextFileMixin):
	''' Loads the transactions line by line, Nx1 frame. '''
	default_path:_ClassVar[_Path] = _Q1D.Groceries
	def __init__(self)->None:
		super().__init__(self.default_path)
		self.load()

class GroceriesProcessed(_DatasetCSV):
	''' The Frequent Item set, hot encoded with pd get_dummies. '''
	default_path:_ClassVar[_Path] = _Q1D.PreProcessed
	def __init__(self,
				grocieres:_Optional[GroceriesDataset]=None
			)->None:
		super().__init__(path=self.default_path)
		if self.path is not None and self.path.exists():
			self.frame = self.load()
		elif grocieres is not None:
			self.frame = grocieres.get_frame()['Transaction'].str.get_dummies(sep=',').astype(bool)
		else:
			raise Exception('if the path does not exist, groceries must be defined.')

class FrequentItemSet(_DatasetCSV):
	default_path:_ClassVar[_Path] = _Q1D.FrequentItemSet
	def __init__(self,
				gProcessed:_Optional[GroceriesProcessed]=None
			)->None:
		super().__init__(path=self.default_path)
		if self.path is not None and self.path.exists():
			self.frame = self.load()
		elif gProcessed is not None:
			self.frame = _apriori(gProcessed.get_frame(), min_support=0.01, use_colnames=True)
		else:
			raise Exception('path is invalid & gProcessed is None.')
		
	def print_top_ten(self):
		freqset:_pd.DataFrame = self.get_frame()
		top_ten = freqset.sort_values(by='support',ascending=False).iloc[0:10]
		msg:str = 'Question 1.B - Top Ten Frequent Itemsets by Support:'
		sep:str = len(msg)*'-'
		print(msg,sep,sep='\n')
		for rec in top_ten.itertuples(index=False):
			print(f'Support: {rec[0]:.3f}\t{rec[1]}')
		print(sep)

class AssociationRules(_DatasetCSV):
	default_path:_ClassVar[_Path] = _Q1D.AssociatonRules
	
	def __init__(self,
			  	freqSet: _Optional[FrequentItemSet] = None
			  ) -> None:
		super().__init__(path=self.default_path)
		if self.path is not None and self.path.exists():
			self.frame = self.load()
		elif freqSet is not None:
			self.frame = _association_rules(
				freqSet.get_frame(),
				metric='confidence',
				min_threshold=0.2,
			)
		else:
			raise Exception('path is invalid and freqSet is None')

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
		for rec in self.get_frame()[[m.value for m in self.cols]].itertuples(index=False):
			output:_List[str] = []
			for col in self.cols:
				output.append(f'{col.value}: {getattr(rec,col.value)}')
			print(' | '.join(output))
		print(sep)
	
	def recommend(self, transaction:_List[str])->_List[str]:
		''' Given a set of items, recommends another set of items'''
		itemset = set(transaction)
		rules = self.get_frame()
		return rules.loc[
			(rules[self.cols.antecedents] == itemset), self.cols.consequents
		].to_list()

def question_one():
	groceries = GroceriesDataset()
	preProcessed = GroceriesProcessed(grocieres=groceries)
	frequentItemset = FrequentItemSet(gProcessed=preProcessed)
	assRules = AssociationRules(freqSet=frequentItemset)
	frequentItemset.print_top_ten()
	assRules.print_rules()
	def try_save(dataset:_DatasetCSV):
		try:
			dataset.save()
		except FileExistsError:
			print('question one, files already exist.')
	
	for dataset in [preProcessed, frequentItemset, assRules]:
		try_save(dataset)
	return