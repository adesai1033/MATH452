from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from load_mnist import load
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pca import pca_alg

def plot_class_performance(conf_matrix, kernel, reg):

    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), class_accuracy, color='skyblue')
    plt.xlabel("Digit Class")
    plt.ylabel("Accuracy")
    plt.title(f"Per-Class Performance on MNIST Dataset w {kernel} kernel and regularization {reg}")
    plt.xticks(range(10))
    plt.ylim(0, 1)  
    plt.show()

def svm_alg():
    x, y = load()
    x = pca_alg(x, num_dim=80)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    kernels = ["linear", "poly", "sigmoid", "rbf"]
    regularization_params = [0.1, 1, 10]

    # Store the cross-validated accuracy scores
    results = []

    for kernel in kernels:
        for C in regularization_params:
            svm_clf = SVC(kernel=kernel, C=C)
            #print("svm initialized")
            cv_scores = cross_val_score(svm_clf, x_train, y_train, cv=5, n_jobs=-1)
            mean_cv_score = np.mean(cv_scores)
            #print("cross validated")

            svm_clf.fit(x_train, y_train)
            #print("data fitted")
            y_pred = svm_clf.predict(x_test)
            #print("data predicted")
            test_accuracy = accuracy_score(y_test, y_pred)
            #print("accuracy score computed")
            conf_matrix = confusion_matrix(y_test, y_pred)
            #print("confusion matrix made")
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1score = f1_score(y_test, y_pred, average='weighted')
            # Store the results
            results.append({
                "Kernel": kernel,
                "C": C,
                "Mean CV Accuracy": mean_cv_score,
                "Test Accuracy": test_accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1score,
                "Confusion Matrix": conf_matrix
            })
            #print(f"{kernel} kernel with {C} regularization param evaluated")
   


    return results

def main():
    results = svm_alg()
    for result in results:
        print(f"Kernel: {result['Kernel']}, C: {result['C']}\n "
              f"Mean CV Accuracy: {result['Mean CV Accuracy']:.4f}\n "
              f"Accuracy: {result['Test Accuracy']:.4f}\n "
              f"Precision: {result['Precision']:.4f}\n "
              f"Recall: {result['Recall']:.4f}\n "
              f"F1-Score: {result['F1-Score']:.4f} "
              )
        plt.figure(figsize=(8, 6))
        sns.heatmap(result["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - Kernel: {result['Kernel']}, C: {result['C']}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()
        plot_class_performance(result["Confusion Matrix"], result['Kernel'], result['C'])



if __name__ == '__main__':
    main()
