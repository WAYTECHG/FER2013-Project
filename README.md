# Facial Expression Recognition (FER) using CNN & Transfer Learning (FER2013)

This project performs facial expression recognition using the FER2013 dataset with both a custom CNN and multiple transfer learning models (VGG16, MobileNetV2, ResNet50). It supports evaluation, visualization (Grad-CAM), and testing with real input images using OpenCV face detection.

---

## 📁 Dataset Structure

The project uses the FER2013 dataset organized as follows:

fer2013/
├── train/
│ ├── angry/
│ ├── disgust/
│ ├── fear/
│ ├── happy/
│ ├── neutral/
│ ├── sad/
│ ├── surprise/
├── test/
│ ├── angry/
│ ├── disgust/
│ ├── fear/
│ ├── happy/
│ ├── neutral/
│ ├── sad/
│ ├── surprise/

Each class folder contains emotion images in `.jpg` or `.png` format.

---

## ⚙️ Installation and Setup

Required Python packages:

- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- tf-keras-vis

Install all dependencies using pip:

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn tf-keras-vis

```

## 🔧 Features

This project includes both a custom CNN and transfer learning models. Each model is designed to work with RGB images (48x48x3) and uses a softmax output for emotion classification.
+----------------------------------------------------------------------------------------------------+
| Model         | Type              | Input Shape | Notable Layers                                   |
|---------------|-------------------|-------------|--------------------------------------------------|
| CNN           | Custom            | 48x48x3     | Conv + BN + GAP                                  |
| VGG16         | Transfer Learning | 48x48x3     | VGG16 (frozen + partial fine-tune) + GAP + Dense |
| MobileNetV2   | Transfer Learning | 48x48x3     | MobileNetV2 + GAP + Dense (256→128→7)            |
| ResNet50      | Transfer Learning | 48x48x3     | ResNet50 + GAP + Dense (256→7)                   |
+----------------------------------------------------------------------------------------------------+

---

## 🔍 Evaluation & Visualization Details

Each model in this project is trained and evaluated using a consistent setup to ensure fair comparison. The evaluation pipeline includes not only accuracy but also class-wise performance insights and interpretability.

### Optimizer
- **Adam**

### Callbacks
- **EarlyStopping**
- **ReduceLROnPlateau**
- **ModelCheckpoint**

### 📈 Evaluation Metrics
- **Accuracy** on validation and test sets
- **Classification Report**:
  - Per-class **Precision**, **Recall**, and **F1-Score**
- **Confusion Matrix**:
  - Heatmap visualization showing true vs predicted labels
- **ROC Curve**:
  - One-vs-Rest AUC plot for all 7 emotion classes

### 🔬 Grad-CAM Visualizations
- Uses **tf-keras-vis** to generate heatmaps
- Applied on the **penultimate convolutional layer** of each model:
  - CNN: `last_conv`
  - VGG16: `block5_conv3`
  - ResNet50: `conv4_block6_out`
  - MobileNetV2: `Conv_1`
- Outputs:
  - Side-by-side comparison of the original image, Grad-CAM overlay, and class confidence scores
  - Grid view for inspecting multiple test samples simultaneously

---

## 📚 References

- [1] **FER2013 Dataset**  
  Kaggle. “Facial Expression Recognition 2013.”  
  https://www.kaggle.com/datasets/msambare/fer2013
  
- [2] **Grad-CAM (Keras Example)**
  Keras Team. *Class Activation Heatmap (Grad-CAM)*. 
  https://keras.io/examples/vision/grad_cam/

- [3] **tf-keras-vis**  
  tf-keras-vis: Keras model visualization toolkit  
  https://github.com/keisen/tf-keras-vis

- [4] **Keras Applications Documentation**  
  https://keras.io/api/applications/

- [5] **scikit-learn Metrics Documentation**  
  https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics