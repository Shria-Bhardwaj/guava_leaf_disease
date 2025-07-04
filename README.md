# 🍃 Guava Leaf Disease Classification using CNN

🚀 A deep learning project focused on identifying **five types of diseases in guava leaves** using a **Convolutional Neural Network (CNN)**, built and trained in **TensorFlow/Keras**.  
This project leverages real image data, data augmentation, and model persistence to classify leaf conditions with high accuracy.

> ✅ **Project Type:** Image Classification  
> 📍 **Framework:** TensorFlow/Keras  
> 🎯 **Goal:** Predict diseases in guava leaves  
> 📁 **Dataset:** Custom image dataset (train, test, val folders in Drive)

---

## 📂 Dataset Details

The dataset is **categorized** into 5 guava leaf conditions. Each category contains colored images of guava leaves with varying lighting and angles to simulate real-world conditions.

### 🔍 Classes

- 🌿 `Canker` – infected leaves showing raised lesions.
- ⚪ `Dot` – minor infections with visible dots or specks.
- ✅ `Healthy` – clean, disease-free leaves.
- 🍂 `Mummification` – dry and crumpled leaf texture.
- 🔥 `Rust` – orange or brown fungal spots on leaves.

### 🗂️ Structure

```
📁 drive/My Drive/
    ├── train/
    │   ├── Canker/
    │   ├── Dot/
    │   ├── Healthy/
    │   ├── Mummification/
    │   └── Rust/
    ├── val/
    │   └── [same structure as train]
    └── test/
        └── [same structure as train]
```

## 🧠 Model Workflow

This project follows an end-to-end pipeline from preprocessing to prediction:

### 1. 🔗 Google Drive Integration

Mount Google Drive to access image folders directly from Colab.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 2. 🖼️ Image Preprocessing

Used `ImageDataGenerator` for loading, rescaling, and augmenting image data.

```python
ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
```

---

### 3. 🏗️ CNN Model Architecture

A simple CNN with 2 convolutional layers, ReLU activation, max-pooling, flattening, and dense output layer with softmax activation for multi-class classification.

```python
model = Sequential([
  Conv2D(32, (3,3), activation='relu'),
  MaxPooling2D(2,2),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D(2,2),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(5, activation='softmax')
])
```

---

### 4. ⚙️ Model Compilation

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

### 5. 🧬 Model Training

Model is trained using `.fit()` on augmented `train_gen` and validated on `val_gen`.

```python
model.fit(train_gen, validation_data=val_gen, epochs=10)
```

---

### 6. 📉 Model Evaluation

Evaluated model performance using `.evaluate()` on test data.

---

### 7. 💾 Model Saving and Loading

```python
model.save('guava_leaf_model.h5')
model.load_weights('guava_leaf_model.h5')
```

---

### 8. 🧪 Prediction on New Data

Used `model.predict()` on new images to test the classification pipeline.

```python
pred = model.predict(image)
predicted_class = class_names[np.argmax(pred)]
```

---

## 📊 Key Insights

- ✅ Achieved high accuracy using a lightweight CNN.
- 🔁 Data augmentation significantly improved model generalization.
- 🧹 Clear folder structure helped streamline image loading and training.
- 💾 Saved models enable reuse without retraining.

---

## 🛠️ Requirements

```bash
tensorflow
numpy
matplotlib
```

> ⚙️ This project was executed on **Google Colab** using a GPU runtime.

---

## 📥🖼️ Interactive Prediction Interface
At the end of the project, a simple and intuitive user interface has been created that allows you to:

1. Upload a guava leaf image
   
![Screenshot 2025-07-04 180102](https://github.com/user-attachments/assets/f02bd389-159d-4b4c-9a79-043b90f8c586)
![Screenshot 2025-07-03 123032](https://github.com/user-attachments/assets/c06730b9-a482-4c53-b160-a345baef3c8c)


3. Automatically classify the disease using the trained CNN model and get an instant prediction output on screen
   
![Screenshot 2025-07-03 123047](https://github.com/user-attachments/assets/4c9edbf4-bbbf-485e-a247-f25f30c02b89)
![Screenshot 2025-07-03 123110](https://github.com/user-attachments/assets/28599604-3b69-4208-93b8-7c48d214cd7d)

📌 This is especially useful for testing the model on custom or unseen data without needing to run any code manually.

---

## 🙋‍♀️ Author

**Shria Bhardwaj**  
🔗 [LinkedIn Profile](www.linkedin.com/in/shria-bhardwaj)

---
