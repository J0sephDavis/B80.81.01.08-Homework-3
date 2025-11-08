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
folder_root_datasets:_Path = _Path('datasets')
class QuestioneOneData:
	logger_name:_ClassVar[str] = f'{app_logger_name}.Q1'
	Groceries:_ClassVar[_Path] = folder_root_datasets.joinpath('q1/groceries.csv') # Raw data
	PreProcessed:_ClassVar[_Path]=folder_root_datasets.joinpath('q1/recommendation_data.csv') # Hot encoded data
	FrequentItemSet:_ClassVar[_Path]=folder_root_datasets.joinpath('q1/frequent_item_set.csv') # Meets support
	AssociatonRules:_ClassVar[_Path]=folder_root_datasets.joinpath('q1/association_rules.csv') # Meets confidence

class QuestionTwoData:
	logger_name:_ClassVar[str] = f'{app_logger_name}.Q2'
	Rankings:_ClassVar[_Path]= folder_root_datasets.joinpath('q2/Universities.csv')
	CleanNormal:_ClassVar[_Path] = folder_root_datasets.joinpath('q2/UCleanNormal.csv')
	CleanNormalMean:_ClassVar[_Path] = folder_root_datasets.joinpath('q2/UCleanNormalMean.csv')
	CleanNormalMedian:_ClassVar[_Path] = folder_root_datasets.joinpath('q2/UCleanNormalMedian.csv')
	folder_dendrograms:_ClassVar[_Path] = _Path('q2/dendrograms')
	folder_boxplots:_ClassVar[_Path] = _Path('q2/boxplots')
	CleanNormalLabeled:_ClassVar[_Path] = folder_root_datasets.joinpath('q2/UCleanNormalLabeled.csv')
	PrivatePublicLabels:_ClassVar[_Path] = folder_root_datasets.joinpath('q2/PublicPrivateSummary.csv')
	StateLabels:_ClassVar[_Path] = folder_root_datasets.joinpath('q2/StateLabelSummary.csv')

class QuestionThreeData:
	logger_name:_ClassVar[str] = f'{app_logger_name}.Q3'
	folder_dendrograms:_ClassVar[_Path] = _Path('q3/dendrograms')
	folder_datasets:_ClassVar[_Path] = folder_root_datasets.joinpath('q3')

	Cereal:_ClassVar[_Path] = folder_datasets.joinpath('Cereals.csv')
	CerealCleanedNormalized:_ClassVar[_Path] = folder_datasets.joinpath('CerealsCleanNormal.csv')
	CerealCleanedNormalLabeled:_ClassVar[_Path] = folder_datasets.joinpath('CerealsCleanNormalLabeled.csv')

class FlierData:
	logger_name:_ClassVar[str] = f'{app_logger_name}.fliers'
	FrequentFliers:_Final[_Path] = folder_root_datasets.joinpath('EastWestAirlinesCluster.csv')
	Normalized:_ClassVar[_Path] = folder_root_datasets.joinpath('Normalized.csv')

class QuestionFourData:
	logger_name:_ClassVar[str] = f'{FlierData.logger_name}.Q4'
	folder_datasets:_ClassVar[_Path] = folder_root_datasets.joinpath('q4')
	folder_figures:_ClassVar[_Path] = _Path('q4')

	LabeledEuclidean:_ClassVar[_Path] = folder_datasets.joinpath('LabeledEuclidean.csv')
	LabeledWard:_ClassVar[_Path] = folder_datasets.joinpath('LabeledWard.csv')
	LabeledKMeans:_ClassVar[_Path] = folder_datasets.joinpath('LabeledKMeans.csv')