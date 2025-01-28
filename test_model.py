import numpy as np
import scipy.io as sio
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load EMNIST dataset
emnist_path = 'emnist-byclass.mat'  # Make sure this file is in your project folder
mat = sio.loadmat(emnist_path)
data = mat['dataset']

# Extract testing data and labels
x_test = np.array(data['test'][0][0]['images'][0][0])  # Images
y_test = np.array(data['test'][0][0]['labels'][0][0])  # Labels

# Preprocess the test data
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0  # Reshape and normalize to [0, 1]

# One-hot encode the labels
num_classes = 62  # EMNIST has 62 classes (10 digits + 26 uppercase + 26 lowercase)
y_test = to_categorical(y_test, num_classes=num_classes)

# Load the saved model
model = load_model('handwriting_recognition_model.h5')

# Predict and print results for the first 5 test samples
for i in range(5):  # Test first 5 samples
    sample_image = x_test[i]
    sample_image = np.expand_dims(sample_image, axis=0)
    prediction = model.predict(sample_image)
    predicted_class = np.argmax(prediction)
    print(f"Test sample {i} predicted class: {predicted_class}")
    print(f"True class: {np.argmax(y_test[i])}")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Generate predictions and true labels
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes))
model.save("final_handwriting_recognition_model.h5")
