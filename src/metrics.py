"""
Task 5.2 – Metric functions (written from scratch, no scikit-learn).

Functions:
    accuracy(y_true, y_pred)        -> float
    confusion_matrix(y_true, y_pred, n_classes=5) -> 2D list
    per_class_accuracy(conf_mat)    -> list of floats
"""


def accuracy(y_true, y_pred):
    """Overall accuracy = fraction of correct predictions."""
    assert len(y_true) == len(y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, n_classes=5):
    """
    Confusion matrix C where C[j][k] = number of samples whose true class is j
    and predicted class is k.
    Rows = true classes, columns = predictions.
    """
    mat = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        mat[t][p] += 1
    return mat


def per_class_accuracy(conf_mat):
    """Per-class accuracy: for class j, C[j][j] / sum(C[j][k] for all k)."""
    accs = []
    for j in range(len(conf_mat)):
        row_sum = sum(conf_mat[j])
        if row_sum > 0:
            accs.append(conf_mat[j][j] / row_sum)
        else:
            accs.append(0.0)
    return accs


def balanced_accuracy(conf_mat):
    """Balanced accuracy = mean of per-class accuracies."""
    pca = per_class_accuracy(conf_mat)
    return sum(pca) / len(pca)


if __name__ == "__main__":
    # Quick sanity test
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    print(f"Accuracy: {accuracy(y_true, y_pred)}")
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    print(f"Confusion matrix: {cm}")
    print(f"Per-class accuracy: {per_class_accuracy(cm)}")
    print(f"Balanced accuracy: {balanced_accuracy(cm)}")
