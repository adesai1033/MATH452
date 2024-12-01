from load_mnist import load
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def knn_alg(k, imgs, labels):
    x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def plot_confusion_matrix(conf_matrix, k):
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Optimal k={k}')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, int(conf_matrix[i, j]),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    plt.show()
    plot_class_performance(conf_matrix, k)

def plot_class_performance(conf_matrix, k):
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), class_accuracy, color='skyblue')
    plt.xlabel("Digit Class")
    plt.ylabel("Accuracy")
    plt.title(f"Per-Class Performance on MNIST Dataset with optimal k={k}")
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.show()

def evaluate_k_for_seed(imgs, labels, k_values, seed):
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    scores_for_k = {k: [] for k in k_values}
    precision_for_k = {k: [] for k in k_values}
    recall_for_k = {k: [] for k in k_values}
    f1_for_k = {k: [] for k in k_values}
    confusion_matrices_for_k = {k: [] for k in k_values}

    for train_index, test_index in kf.split(imgs):
        x_train, x_test = imgs[train_index], imgs[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            scores_for_k[k].append(accuracy)
            precision_for_k[k].append(precision)
            recall_for_k[k].append(recall)
            f1_for_k[k].append(f1)
            confusion_matrices_for_k[k].append(conf_matrix)

    avg_scores_for_k = {k: np.mean(scores) for k, scores in scores_for_k.items()}
    avg_precision_for_k = {k: np.mean(scores) for k, scores in precision_for_k.items()}
    avg_recall_for_k = {k: np.mean(scores) for k, scores in recall_for_k.items()}
    avg_f1_for_k = {k: np.mean(scores) for k, scores in f1_for_k.items()}
    avg_conf_matrices_for_k = {k: np.mean(conf_matrices, axis=0) for k, conf_matrices in confusion_matrices_for_k.items()}
    
    return avg_scores_for_k, avg_precision_for_k, avg_recall_for_k, avg_f1_for_k, avg_conf_matrices_for_k

def find_optimal_k(k_values, num_repeats=10):
    imgs, labels = load()
    best_k = None
    best_score = 0
    all_scores = {k: [] for k in k_values}
    all_precision = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    all_f1 = {k: [] for k in k_values}
    avg_conf_matrices_for_k = {k: None for k in k_values}

    for _ in range(num_repeats):
        seed = randint(0, 10000)
        avg_scores_for_k, avg_precision_for_k, avg_recall_for_k, avg_f1_for_k, conf_matrices_for_k = evaluate_k_for_seed(imgs, labels, k_values, seed)

        for k, score in avg_scores_for_k.items():
            all_scores[k].append(score)
            all_precision[k].append(avg_precision_for_k[k])
            all_recall[k].append(avg_recall_for_k[k])
            all_f1[k].append(avg_f1_for_k[k])
            if avg_conf_matrices_for_k[k] is None:
                avg_conf_matrices_for_k[k] = conf_matrices_for_k[k]
            else:
                avg_conf_matrices_for_k[k] += conf_matrices_for_k[k]

    avg_scores_across_seeds = {k: np.mean(scores) for k, scores in all_scores.items()}
    avg_precision_across_seeds = {k: np.mean(prec) for k, prec in all_precision.items()}
    avg_recall_across_seeds = {k: np.mean(rec) for k, rec in all_recall.items()}
    avg_f1_across_seeds = {k: np.mean(f1s) for k, f1s in all_f1.items()}
    avg_conf_matrices_across_seeds = {k: matrix / num_repeats for k, matrix in avg_conf_matrices_for_k.items()}

    for k, avg_score in avg_scores_across_seeds.items():
        print(f"Overall average accuracy for k={k}: {avg_score}")
        print(f"Precision: {avg_precision_across_seeds[k]}, Recall: {avg_recall_across_seeds[k]}, F1 Score: {avg_f1_across_seeds[k]}")
        if avg_score > best_score:
            best_score = avg_score
            best_k = k

    print(f"Optimal k={best_k} metrics\nAccuracy:{best_score}\nPrecision: {avg_precision_across_seeds[best_k]}\nRecall: {avg_recall_across_seeds[best_k]}\nF1 Score: {avg_f1_across_seeds[best_k]}")
    plot_confusion_matrix(avg_conf_matrices_across_seeds[best_k], best_k)
    return best_k, best_score

def main():
    k_values = range(1, 10)
    optimal_k, best_score = find_optimal_k(k_values)
    print(f"Optimal k: {optimal_k} with cross-validation accuracy: {best_score}")

if __name__ == '__main__':
    main()
