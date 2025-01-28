from tensorflow.keras.models import load_model

# Load the trained model (ensure the file exists from your previous training)
model = load_model('handwriting_recognition_model.h5')  # Replace with the correct model path

# Save it again or perform additional operations
model.save('handwriting_recognition_model.h5')
print("Model saved as handwriting_recognition_model.h5")
