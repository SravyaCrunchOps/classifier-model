from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) # 100%

    cr = classification_report(y_test, y_pred)
    # print("calssification report: ", cr)

    cm = confusion_matrix(y_test, y_pred)
    # print("confusion matrix report: ", cm)

    return accuracy