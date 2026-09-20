from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def build_model():
    model = Sequential([
        Input(shape=(784,)),
        Dense(128, activation='relu', name='L1'),
        Dense(64, activation='relu', name='L2'),
        Dense(10, name='Output')
    ])
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(),
        metrics=["accuracy"]
    )
    return model