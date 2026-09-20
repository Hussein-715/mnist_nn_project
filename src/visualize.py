import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_curves(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["accuracy"], label="Training_Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation_Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("Images/training_curves/Training_vs_Validation_Accuracy.png")
    plt.show()
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training_Loss")
    plt.plot(history.history["val_loss"], label="Validation_Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("Images/training_curves/Training_vs_Validation_Loss.png")
    plt.show()

def plot_confusion_matrix(y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10)
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("MNIST Confusion Matrix")
    plt.savefig("Images/Confusion_Matrix.png")
    plt.show()

def plot_misclassified(X_test,y_test,y_pred,n):
    wrong_indices = np.where(y_pred != y_test)[0]
    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i, ax in enumerate(axes):
        index = wrong_indices[i]
        ax.imshow(X_test[index].reshape(28, 28), cmap='gray')
        ax.set_title(f"Prediction:{y_pred[index]}\nActual  : {y_test[index]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("Images/Misclassified.png")
    plt.show()