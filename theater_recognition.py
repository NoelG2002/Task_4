import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input

haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(haar_cascade_path):
    raise FileNotFoundError(f"Haar Cascade file not found at {haar_cascade_path}")
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
# Path to your dataset
image_paths = glob('theater_recognition/utkface_aligned_cropped/UTKFace/*.jpg')
age_labels = []

# Count images per age
age_count = defaultdict(int)

for path in image_paths:
    filename = os.path.basename(path)
    if filename.endswith('.chip'):
        filename = filename[:-5]

    try:
        age = int(filename.split('_')[0])
        age_labels.append(age)
        age_count[age] += 1
    except ValueError:
        print(f"Skipping file due to unexpected format: {filename}")

# Print the age distribution
print("Age distribution:", dict(age_count))

def create_model():
    model = models.Sequential([
        Input(shape=(48, 48, 3)),  # Define the input shape here
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')  # Output age as a continuous value
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Load and preprocess images
def preprocess_images(image_paths, target_size=(48, 48)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0  # Normalize to [0, 1] range
            images.append(img)
    return np.array(images)

images = preprocess_images(image_paths)
age_labels = np.array(age_labels)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(images, age_labels, test_size=0.2, random_state=42)

# Create or load the model
model_path = 'age_detection_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Loaded saved model.")
else:
    model = create_model()
    print("Created a new model.")

# Train the model if not already trained
if not os.path.exists(model_path):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

    # Save the model
    model.save(model_path)
    print("Model saved as 'age_detection_model.h5'.")
else:
    print("Model already trained.")

def predict_age(frame):
    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0  # Normalize
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        
        # Predict age
        predicted_age = model.predict(face, verbose=0)  # Set verbose=0 to suppress output
        return int(predicted_age[0][0])  # Convert to integer

    return None  # No face detected

# Start video capture
cap = cv2.VideoCapture(0)

data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    age = predict_age(frame)
    entry_time = pd.Timestamp.now()

    if age is not None:
        print(f"Predicted age: {age}")
        if age < 13 or age > 60:
            cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 255), 2)  # Red rectangle
            cv2.putText(frame, "Not allowed", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            data.append({"age": age, "entry_time": entry_time})
        else:
            cv2.putText(frame, f"Age: {age}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            data.append({"age": age, "entry_time": entry_time})
    else:
        cv2.putText(frame, "No face detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Age Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the entry log to CSV
df = pd.DataFrame(data)
df.to_csv("entry_log.csv", index=False)
print("Entry log saved as 'entry_log.csv'.")
