import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp = np.count_nonzero(np.logical_and(prediction, ground_truth))
    fp = np.count_nonzero(np.logical_and(prediction, np.logical_not(ground_truth)))
    fn = np.count_nonzero(np.logical_and(np.logical_not(prediction), ground_truth))
    tn = np.count_nonzero(np.logical_and(np.logical_not(prediction), np.logical_not(ground_truth)))
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    if (tp + tn + fp + fn) != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    if (fp + fn) != 0:
        f1 = tp / (tp + 0.5 * (fp + fn))
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    tp = np.count_nonzero(np.logical_and(prediction, ground_truth))
    fp = np.count_nonzero(np.logical_and(prediction, np.logical_not(ground_truth)))
    fn = np.count_nonzero(np.logical_and(np.logical_not(prediction), ground_truth))
    tn = np.count_nonzero(np.logical_and(np.logical_not(prediction), np.logical_not(ground_truth)))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy
