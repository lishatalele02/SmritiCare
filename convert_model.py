import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("cnn_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# Save the model
with open("cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion completed successfully!")