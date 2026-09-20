def train_model(model,X_train,y_train):
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_split=0.1
    )
    model.save("models/mnist_digit_model.keras")
    return history