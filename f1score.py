import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	Y_true = np.array(y_true)
	Y_pred = np.array(y_pred)
	
	true_positive = 0
	false_positive = 0
	false_negative = 0
	true_negative = 0
	
	precision = 0
	recall = 0
	

	for grou_y, pred_y in zip(Y_true, Y_pred):
		if grou_y == 1:
			if pred_y == 1:
				true_positive += 1
			else:
				false_positive += 1
		else:
			if pred_y == 1:
				false_negative += 1
			else:
				true_negative += 1

	precision = true_positive / (true_positive + false_positive)if(true_positive + false_positive)!= 0 else 0
	recall = true_positive / (true_positive + false_negative)if(true_positive + false_negative)!= 0 else 0
	f1 = 2 * (precision * recall) / (precision + recall)if(precision + recall)!=0 else 2 * (precision * recall)
	
	f1_sklearn = f1_score(Y_true, Y_pred)
	precision_sklearn = precision_score(Y_true, Y_pred)
	recall_sklearn = recall_score(Y_true, Y_pred)

	return round(f1,3), round(f1_sklearn,3), round(precision,3), round(recall,3),round(precision_sklearn,3),round(recall_sklearn,3)
	
if __name__ == "__main__":
	y_true = list(map(int, input("Y_true: ").split(",")))
	y_pred = list(map(int, input("Y_pred: ").split(",")))
	print(calculate_f1_score(y_true, y_pred))
	
	
	
