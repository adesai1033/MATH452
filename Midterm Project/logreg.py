
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')
from load_mnist import load

#test if mnist data loaded properly
def plot_digit_samples(X, y, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):

        idx = np.random.randint(0, len(X))

        plt.subplot(1, num_samples, i + 1)

        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Digit: {y[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def log_reg(x_train, y_train, x_test, y_test):
    model = Pipeline([('scaler', 
            StandardScaler()),
            ('logistic', 
            LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000,random_state=42))])

    print("Training the model... This might take a few minutes...")
    model.fit(x_train, y_train)
    print("Training complete!")
    y_pred = model.predict(x_test)


    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print("\nDetailed Performance Report:")
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.show()
    return y_pred

def plot_predictions(X, y_true, y_pred, num_samples=5):

    indices = np.random.randint(0, len(X), num_samples)

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_true[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_accuracy_by_class(y_test, y_pred):

    per_digit_accuracy = {}
    for digit in np.unique(y_test):
        mask = (y_test == digit)
        digit_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        per_digit_accuracy[digit] = digit_accuracy


    plt.figure(figsize=(12, 6))
    digits = list(per_digit_accuracy.keys())
    accuracies = list(per_digit_accuracy.values())

    plt.bar(digits, accuracies)
    plt.title('Accuracy by Digit')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)  

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')

    plt.show()

    print("\nAccuracy for each digit:")
    for digit, acc in per_digit_accuracy.items():
        print(f"Digit {digit}: {acc:.3f}")

def main():
    x, y = load()
    plot_digit_samples(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = log_reg(x_train, y_train, x_test, y_test)
    plot_predictions(x_test, y_test, y_pred)
    plot_accuracy_by_class(y_test, y_pred)

if __name__ == '__main__':
    main()
