# 🐶 Dog Vision - Dog Breed Classifier using Keras & TensorFlow Hub

This project demonstrates a simple yet powerful deep learning pipeline to classify images of dogs using transfer learning and the Keras API. The model is built and trained in **Google Colab**, utilizing **TensorFlow**, **Keras**, and **TensorFlow Hub** for fast and effective image classification.

## 🚀 Project Objective

To develop a convolutional neural network (CNN)-based model that accurately predicts the breed of a dog given its image. This is achieved by leveraging a pretrained model from **TensorFlow Hub** and fine-tuning it on a smaller dog image dataset.

## 🛠️ Technologies Used

- **Python 3.11**
- **TensorFlow 2.x**
- **Keras** (via `tf.keras`)
- **TensorFlow Hub** – for loading pretrained image models (e.g., `mobilenet_v2`)
- **NumPy & Matplotlib** – for data handling and visualization
- **Google Colab** – for training on GPUs

## 🧠 Model Architecture

- A **pretrained MobileNetV2** model is used as a feature extractor via `hub.KerasLayer`.
- A **Dense output layer** with softmax activation classifies the input into the defined number of dog breeds.
- The model is compiled using **categorical crossentropy** loss and **Adam** optimizer.

### Code Snippet:
hub_layer = hub.KerasLayer(MODEL_URL, input_shape=(224, 224, 3), trainable=False)

model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(num_classes, activation="softmax")
])
📂 Dataset
The training and testing datasets consist of labeled images of various dog breeds. Images are resized to 224x224 to match the input requirements of the pretrained model. The dataset is preprocessed and loaded using Keras' ImageDataGenerator.

📈 Training
The model is trained for a few epochs, tracking training and validation accuracy. Overfitting is monitored, and model performance is visualized using accuracy and loss curves.

🧪 Evaluation & Prediction
After training, the model is evaluated on a test dataset and used to predict the breed of new, unseen images.

test_predictions = loaded_model.predict(test_data, verbose=1)
🐞 Common Issues
If you encounter the error 'dict' object is not callable, ensure you're using hub.KerasLayer() instead of hub.load() when defining the model.

When loading a model saved with a custom layer (KerasLayer), use:


load_model("path", custom_objects={"KerasLayer": hub.KerasLayer})
📌 Future Plans
Expand the model to recognize more dog breeds

Add Gradio/Streamlit UI for image upload and prediction

Improve generalization with data augmentation

🙌 Acknowledgements
TensorFlow Hub

Keras Documentation

Google Colab for providing free GPU support
