from data_loader import load_emnist
from model import create_model
print("create_model imported successfully!")
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load the dataset
(x_train, y_train), (x_test, y_test) = load_emnist()

# Create the model
model = create_model()

# Save the best model
checkpoint = ModelCheckpoint('handwriting_model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,  # Adjust epochs if needed
    batch_size=128,  # Adjust batch size if needed
    callbacks=[checkpoint]
)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
