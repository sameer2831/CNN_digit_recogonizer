# 🧠 Handwritten Digit Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** for recognizing handwritten digits using a Kaggle dataset (not MNIST). It uses two separate datasets: one labeled (`train.csv`) and one unlabeled (`test.csv`), and is fully compatible with Google Colab.

---

## 📁 Dataset Structure

- `train.csv` – Contains labeled data with 785 columns (`label` + 784 pixel values).
- `test.csv` – Contains only pixel values (no labels), used for prediction submission.

Each image is 28x28 pixels, flattened into a row of 784 values.

---

## 🚀 Objective

To build, train, and evaluate a CNN that classifies digits from 0 to 9 using labeled image data, and generate predictions for unseen test data.

---

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Scikit-learn
- Matplotlib / Seaborn

---

## 🧼 Data Preprocessing

- Normalized pixel values from `[0, 255]` → `[0, 1]`
- Reshaped flat images to `28x28x1` format for CNN input
- One-hot encoded the labels
- Split the training set into 90% train and 10% validation

```python
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)
y_cat = to_categorical(y, 10)
```

---

## 🧠 Model Architecture

``` bash
Input: 28x28x1 Grayscale Image

1. Conv2D (32 filters, 3x3, ReLU)
2. MaxPooling2D (2x2)
3. Conv2D (64 filters, 3x3, ReLU)
4. MaxPooling2D (2x2)
5. Flatten
6. Dense (128 units, ReLU)
7. Dropout (0.3)
8. Dense (10 units, Softmax)
```
```
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
## 📊 Visualizations
### 🖼️ Sample Digits

- 📈 Accuracy and Loss
  
 Training and validation accuracy/loss over epochs are plotted to evaluate convergence and overfitting.

### 📌 Label Distribution
  - Bar plot showing how digits are distributed in the training set.

### 📉 Confusion Matrix (on validation set)
  - Reveals which digits the model struggles with the most.

### ❌ Misclassified Examples
  - Visuals of digits that were incorrectly predicted, showing true and predicted labels.

### 🧪 Evaluation Metrics
  - **Accuracy**: Achieved over 98% on validation set.
  - **Loss**: Stable convergence, indicating no overfitting.
  - **Confusion Matrix**: Shows strong diagonal indicating good predictions.
  - **Classification Report**: Precision, recall, and F1-score across digits.

## 🧾 Prediction & Submission
```
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
```
## 📁 Folder Structure
```
📂 digit-recognition-cnn
├── train.csv
├── test.csv
├── CNN_digit_recogonizer.ipynb
├── submission.csv
└── README.md
```

## 💡 Future Improvements

- Implement data augmentation using ImageDataGenerator

- Hyperparameter tuning

- Try different CNN architectures (ResNet, VGG)

- Deploy model using Flask or Streamlit

## 🤝 Author

- Sameer – MS Computer Science, Syracuse University

## 📌 References
- Kaggle: Handwritten Digits Dataset

- TensorFlow Keras Documentation



## ✅ License

This project is licensed under the MIT License.

---

Let me know if you'd like this markdown converted into a downloadable file, or auto-uploaded to GitHub, or enhanced with image links!







