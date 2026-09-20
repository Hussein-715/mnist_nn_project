import numpy as np
import tensorflow as tf

def evaluate_model(model,X_test,y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    logits = model.predict(X_test)
    probabilities = tf.nn.softmax(logits)
    y_pred = np.argmax(probabilities, axis=1)
    return test_loss,test_accuracy,y_pred