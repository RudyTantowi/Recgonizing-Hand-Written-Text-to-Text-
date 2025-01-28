from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical

# Fungsi untuk membuat model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(62, activation='softmax')  # 62 classes (0-9, A-Z, a-z)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Scheduler untuk learning rate
def scheduler(epoch, lr):
    if epoch < 10:
        return lr  # Tetap sama untuk 10 epoch pertama
    else:
        return lr * 0.1  # Kurangi learning rate setelah 10 epoch

# Load EMNIST dataset
emnist_path = 'emnist-byclass.mat'  # Pastikan file ada di folder proyek
mat = sio.loadmat(emnist_path)
data = mat['dataset']

# Ekstrak data pelatihan
x_train = np.array(data['train'][0][0]['images'][0][0])  # Gambar
y_train = np.array(data['train'][0][0]['labels'][0][0])  # Label

# Preprocessing data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # Normalisasi ke [0, 1]

# One-hot encode label
num_classes = 62
y_train = to_categorical(y_train, num_classes=num_classes)

# Buat model
model = create_model()

# Callback untuk learning rate scheduler
lr_scheduler = LearningRateScheduler(scheduler)

# Latih model
model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    callbacks=[lr_scheduler]  # Tambahkan scheduler di sini
)

# Simpan model
model.save('handwriting_recognition_model.h5')
