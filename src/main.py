from data import load_data
from model import build_model
from train import train_model
from evaluate import evaluate_model
from visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_misclassified
)

X_train, y_train, X_test, y_test = load_data()
model = build_model()
history = train_model(model, X_train, y_train)
plot_training_curves(history)
test_loss,test_accuracy,y_pred = evaluate_model(model, X_test, y_test)
print("Test Loss : ",test_loss)
print("Test Accuracy : ",test_accuracy)
plot_confusion_matrix(y_test, y_pred)
plot_misclassified(X_test,y_test,y_pred,10)





