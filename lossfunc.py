import numpy as np

def focal_loss(y_true, y_pred, alpha, gamma):
    """
    Focal loss is a loss function used in object detection tasks.
    It is a variant of the cross-entropy loss that is designed to address the problem of class imbalance.
    It is defined as:
    L(y, p) = -alpha * (1 - p)^gamma * log(p)
    where y is the true label, p is the predicted probability, alpha is the weighting factor, and gamma is the focusing parameter.
    """
    epsilon = 1e-7
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    p = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -alpha * (1 - p) ** gamma * np.log(p)
    loss = np.sum(loss)
    return loss

def triplet_loss(anchor, positive, negative, margin):
    """
    Triplet loss is a loss function used in triplet network.
    It is defined as:
    L(a, p, n) = max(0, d(a, p) - d(a, n) + margin)
    where a is the anchor, p is the positive, n is the negative, and margin is the margin parameter.
    """
    def euclidean_distance(x1, x2): 
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)

    anchor_positive_distance = euclidean_distance(anchor, positive)
    anchor_negative_distance = euclidean_distance(anchor, negative)
    loss = np.maximum(0, anchor_positive_distance - anchor_negative_distance + margin)
    loss = np.sum(loss)
    return loss

def binary_crossentropy(y_true, y_pred):
    """
    Binary crossentropy is a loss function used in binary classification tasks.
    It is defined as:
    L(y, p) = -[y*log(p) + (1-y)*log(1-p)]
    where y is the true label and p is the predicted probability.
    """
    epsilon = 1e-7
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    loss = np.mean(loss)
    return loss

def dice_loss(y_true, y_pred, epsilon):
    """
    Dice loss is a loss function used in image segmentation tasks.
    It is defined as:
    L(y, p) = 1 - (2 * intersection + epsilon) / (union + epsilon)
    where y is the true label and p is the predicted probability.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # intersection = y_true * y_pred
    intersection = np.sum(y_true * y_pred)
    # union = y_true + y_pred
    union = np.sum(y_true) + np.sum(y_pred)
    loss = 1 - (2 * intersection + epsilon) / (union + epsilon)
    return loss

if __name__ == "__main__":
    # Binary classification data
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0.1, 0.9, 0.2, 0.8, 0.3]
    
    # Triplet data (anchor, positive, negative examples)
    anchor = [0.2, 0.8, 0.3, 0.7, 0.4]
    
    # Parameters
    alpha = 0.5
    gamma = 2
    margin = 1
    epsilon = 1e-7
    # Test all loss functions
    print("Focal Loss:", focal_loss(y_true, y_pred, alpha, gamma))
    print("Triplet Loss:", triplet_loss(anchor, y_true, y_pred, margin))
    print("Binary Crossentropy:", binary_crossentropy(y_true, y_pred))
    print("Dice Loss:", dice_loss(y_true, y_pred, epsilon))
	
	
	
	