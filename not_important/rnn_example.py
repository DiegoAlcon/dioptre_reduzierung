import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate some example data (you can replace this with your own dataset)
input_data = np.random.rand(100, 1)
output_data = np.random.rand(100, 1)

# Create and train the model (same as before)
model = Sequential()
model.add(LSTM(64, input_shape=(None, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(input_data, output_data, epochs=10, batch_size=32)

# Make predictions (you can replace this with your own test data)
test_input = np.random.rand(10, 1)
predictions = model.predict(test_input)

# Generate some random true values for comparison
true_values = np.random.rand(10, 1)

# Plot predictions vs. true values
plt.figure(figsize=(8, 6))
plt.scatter(true_values, predictions, label='Predictions', color='b', marker='o')
plt.plot([0, 1], [0, 1], color='r', linestyle='--', label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs. True Values')
plt.grid(True)
plt.legend()
plt.show()
