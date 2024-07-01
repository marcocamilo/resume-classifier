from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def classifier_report(data: list, classifier, labels: list, graph: bool = True):
    """
    This function takes in a list of data and a classifier and returns the accuracy score of the classifier.
    The function also generates a confusion matrix if the graph parameter is set to True.

    Parameters:
    - data: list: A list of data containing the training and testing data.
    - classifier: object: A classifier object that will be used to train and predict the data.
    - labels: list: A list of labels that will be used to label the confusion matrix.
    - graph: bool: A boolean value that determines if a confusion matrix will be generated.

    Returns:
    accuracy score, heatmap of confusion matrix.
    """
    clf = classifier
    clf.fit(data[0], data[2])
    predictions = clf.predict(data[1])
    accuracy = round(accuracy_score(predictions, data[3])*100, 2) 
    print(f"SVC accruacy score {accuracy}%")

    if graph:
        matrix = confusion_matrix(data[3], predictions)
        ax = sns.heatmap(
            matrix, 
            annot=True, 
            fmt='d', 
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.show()

    return accuracy
