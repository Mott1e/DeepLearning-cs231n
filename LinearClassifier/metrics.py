def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(ground_truth)):
        if prediction[i] == True and ground_truth[i] == True:
            tp+=1
        if prediction[i] == True and ground_truth[i] == False:
            fp+=1
        if prediction[i] == False and ground_truth[i] == False:
            tn+=1
        if prediction[i] == False and ground_truth[i] == True:
            fn+=1

    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        precision, recall, f1 = 0, 0, 0

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
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
    rigth = 0

    for i in range(len(ground_truth)):
        if prediction[i] == ground_truth[i]:
            rigth += 1

    return rigth/len(ground_truth)
