from groceries import *
from recommendation.dataset import RecommendationDataset
groceries = GroceriesDataset()

def question_one():
	recset = RecommendationDataset.from_grocery_data(groceries) # For Question One.
	frequent_items = recset.get_frequent_itemsets_apriori()
	return
question_one()
exit()