from typing import (
	Optional as _Optional,
	List as _List,
	ClassVar as _ClassVar,
)
import pandas as _pd
from helpers.dataset import (
	DatasetBase as _DatasetBase,
	DatasetCSV as _DatasetCSV,
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
import logging as _logging
logger = _logging.getLogger(_Q1D.logger_name)

class GroceriesDataset(_DatasetBase, _DatasetTextFileMixin):
	''' Loads the transactions line by line, Nx1 frame. '''
	default_path:_ClassVar[_Path] = _Q1D.Groceries
	logger.debug(f'GroceriesDataset.default_path={default_path}')
	def __init__(self)->None:
		logger.debug('GroceriesDataset.__init__')
		super().__init__(self.default_path)
		self.load()

class GroceriesProcessed(_DatasetCSV):
	''' The Frequent Item set, hot encoded with pd get_dummies. '''
	default_path:_ClassVar[_Path] = _Q1D.PreProcessed
	logger.debug(f'GroceriesProcessed.default_path={default_path}')
	def __init__(self,
				groceries:_Optional[GroceriesDataset]=None
			)->None:
		logger.debug(f'GroceriesProcessed.__init__(groceries:{groceries})')
		super().__init__(path=self.default_path)
		if self.path is not None and self.path.exists():
			logger.debug('GroceriesProcessed.__init__ load from file')
			self.load()
		elif groceries is not None:
			logger.debug('GroceriesProcessed.__init__ create from groceries')
			self.frame = groceries.get_frame()['Transaction'].str.get_dummies(sep=',').astype(bool)
		else:
			err = Exception('if the path does not exist, groceries must be defined.')
			logger.error('GroceriesProcessed, unreachable code reached.',exc_info=err)
			raise err

class FrequentItemSet(_DatasetBase, _DatasetSaveMixin):
	default_path:_ClassVar[_Path] = _Q1D.FrequentItemSet
	logger.debug(f'FrequentItemSet.default_path={default_path}')
	def __init__(self,
				gProcessed:GroceriesProcessed
			)->None:
		logger.debug(f'FrequentItemSet.__init__(gProcessed:{gProcessed})')
		super().__init__(path=self.default_path)
		logger.debug('FrequentItemSet.__init__ generating apriori frame')
		self.frame = _apriori(gProcessed.get_frame(), min_support=0.01, use_colnames=True)
		
	def print_top_ten(self):
		logger.debug('FrequentItemSet.print_top_ten')
		freqset:_pd.DataFrame = self.get_frame()
		top_ten = freqset.sort_values(by='support',ascending=False).iloc[0:10]
		msg:str = 'Question 1.B - Top Ten Frequent Itemsets by Support:'
		sep:str = len(msg)*'-'
		output:_List[str] = [msg,sep]
		for rec in top_ten.itertuples(index=False):
			output.append(f'Support: {rec[0]:.3f}\t{rec[1]}')
		output.append(sep)
		logger.info('\n'.join(output))

	def save(self, clobber: bool = True, include_index: bool = False):
		if self.frame is not None:
			self.frame.sort_values(by='support',ascending=False, inplace=True)
		return super().save(clobber=clobber,include_index=include_index)

class AssociationRules(_DatasetBase, _DatasetSaveMixin):
	default_path:_ClassVar[_Path] = _Q1D.AssociatonRules
	logger.debug(f'AssociationRules.default_path={default_path}')
	
	def __init__(self,
			  	freqSet: FrequentItemSet
			  ) -> None:
		logger.debug(f'AssociationRules.__init__(freqSet:{freqSet})')
		super().__init__(path=self.default_path)
		logger.debug('AssociationRules.__init__ create from freqSet')
		self.frame = _association_rules(
			freqSet.get_frame(),
			metric='confidence',
			min_threshold=0.2,
		).sort_values('confidence')

	class cols(_StrEnum):
		''' Columns we want to print'''
		antecedents=_auto()
		consequents=_auto()
		support=_auto()
		confidence=_auto()
		lift=_auto()

	def print_rules(self):
		logger.debug('AssociationRules.print_rules')
		''' Print the association rules '''
		msg:str = 'Question 1.C - Association Rules'
		sep:str = len(msg)*'-'
		output:_List[str] = [msg,sep]
		for rec in self.get_frame()[[m.value for m in self.cols]].itertuples(index=False):
			sub_output:_List[str] = []
			for col in self.cols:
				sub_output.append(f'{col.value}: {getattr(rec,col.value)}')
			output.append(' | '.join(sub_output))
		output.append(sep)
		logger.info('\n'.join(output))
	
	def recommend(self, transaction:_List[str])->_List[str]:
		''' Given a set of items, recommends another set of items'''
		logger.debug(f'AssociationRules.recommend(transaction={transaction})')
		itemset = set(transaction)
		logger.debug(f'itemset:{itemset}')
		rules = self.get_frame()
		rv = rules.loc[
			(rules[self.cols.antecedents] == itemset), self.cols.consequents
		].to_list()
		logger.debug(f'recommend->{rv}')
		return rv

def question_one():
	logger.info('===== QUESTION ONE ======')
	groceries = GroceriesDataset()
	preProcessed = GroceriesProcessed(groceries=groceries)
	frequentItemset = FrequentItemSet(gProcessed=preProcessed)
	assRules = AssociationRules(freqSet=frequentItemset)
	frequentItemset.print_top_ten()
	assRules.print_rules()
	
	request = ['soda', 'yogurt']
	logger.info(f'Calculate recommendation for: {request}')
	result = assRules.recommend(request)
	logger.info(f'Recommended item: {result}')

	def try_save(dataset:_DatasetCSV):
		logger.debug(f'Q1.try_save {dataset}')
		try:
			dataset.save()
		except FileExistsError:
			logger.warning('Could not save dataset, already exists.')
			print('question one, files already exist.')
	
	for dataset in [preProcessed, assRules, frequentItemset]:
		try_save(dataset)
	logger.info('=========================')
	return