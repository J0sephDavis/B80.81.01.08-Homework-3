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
app_logger_name:_Final[str] = 'HW3'
class QuestioneOneData:
	logger_name:_ClassVar[str] = f'{app_logger_name}.Q1'
	Groceries:_ClassVar[_Path] = _Path('datasets/q1/groceries.csv') # Raw data
	PreProcessed:_ClassVar[_Path]=_Path('datasets/q1/recommendation_data.csv') # Hot encoded data
	FrequentItemSet:_ClassVar[_Path]=_Path('datasets/q1/frequent_item_set.csv') # Meets support
	AssociatonRules:_ClassVar[_Path]=_Path('datasets/q1/association_rules.csv') # Meets confidence

class QuestionTwoData:
	logger_name:_ClassVar[str] = f'{app_logger_name}.Q2'
	Rankings:_ClassVar[_Path]= _Path('datasets/q2/Universities.csv')
	CleanNormal:_ClassVar[_Path] = _Path('datasets/q2/UCleanNormal.csv')
	CleanNormalMean:_ClassVar[_Path] = _Path('datasets/q2/UCleanNormalMean.csv')
	CleanNormalMedian:_ClassVar[_Path] = _Path('datasets/q2/UCleanNormalMedian.csv')
	folder_dendrograms:_ClassVar[_Path] = _Path('q2/dendrograms')
	folder_boxplots:_ClassVar[_Path] = _Path('q2/boxplots')
	CleanNormalLabeled:_ClassVar[_Path] = _Path('datasets/q2/UCleanNormalLabeled.csv')
	
FILEPATH_EW_AIRLINE:_Final[_Path] = _Path('datasets/EastWestAirlinesCluster.csv')
FILEPATH_CEREAL:_Final[_Path] = _Path('datasets/Cereals.csv')