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