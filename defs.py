from typing import (
	Final as _Final,
	Dict as _Dict,
	ClassVar as _ClassVar
)
from pathlib import Path as _Path

''' Question 1
Groceries -- Raw datasets
Recommendation Data -- The processed dataset
Frequent Item Set -- The frequent itemsets, for assignment results
Association Rules -- The rules for analysis
'''
class QuestioneOneData:
	Groceries:_ClassVar[_Path] = _Path('datasets/q1/groceries.csv') # Raw data
	PreProcessed:_ClassVar[_Path]=_Path('datasets/q1/recommendation_data.csv') # Hot encoded data
	FrequentItemSet:_ClassVar[_Path]=_Path('datasets/q1/frequent_item_set.csv') # Meets support
	AssociatonRules:_ClassVar[_Path]=_Path('dataset/q1/association_rules.csv') # Meets confidence

FILEPATH_UNIVERSITY_RANKINGS:_Final[_Path] = _Path('datasets/Universities.csv')
FILEPATH_EW_AIRLINE:_Final[_Path] = _Path('datasets/EastWestAirlinesCluster.csv')
FILEPATH_CEREAL:_Final[_Path] = _Path('datasets/Cereals.csv')