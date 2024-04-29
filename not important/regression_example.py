import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data for regression
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.randn(100)  # Linear relationship with noise

# Split data into training and validation sets
X_train, y_train = X[:80], y[:80]
X_val, y_val = X[80:], y[80:]

# Create a simple feedforward neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)  # Output layer (single neuron for regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error') # Here Backpropagation is used.

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
X_test = np.linspace(0, 10, 20)
y_pred = model.predict(X_test)

# Visualize predictions
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', label='Predictions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
