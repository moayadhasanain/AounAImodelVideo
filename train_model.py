import h5py
import numpy as np

file_path = r"D:\dataset (video AI)\SumMe.h5"

X = []
y = []

with h5py.File(file_path, "r") as f:
    
    videos = list(f.keys())
    
    for vid in videos:
        features = f[vid]["feature"][:]
        labels = f[vid]["label"][:]
        
        X.append(features)
        y.append(labels)

X = np.concatenate(X)
y = np.concatenate(y)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

X = X.astype("float32")
X = X / np.max(X)


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


model = Sequential([
    Dense(256, activation="relu", input_shape=(512,)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)


loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)


model.save("video_summary_model.h5")