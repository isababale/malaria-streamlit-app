# 🧫 Malaria Cell Classification with Deep Learning

A comprehensive machine learning project for automated detection of malaria-infected blood cells using ResNet18 and Grad-CAM visualization.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/accuracy-97.22%25-brightgreen.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Web Application](#web-application)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements a deep learning solution for automated malaria detection from microscopic blood cell images. Using transfer learning with ResNet18 and advanced visualization techniques (Grad-CAM), the system achieves **97.22% test accuracy** in classifying cells as parasitized or uninfected.

### Key Highlights

- ✅ **High Accuracy**: 97.22% test accuracy
- 🔍 **Interpretable AI**: Grad-CAM visualizations show model decision-making
- 🚀 **Transfer Learning**: Fine-tuned ResNet18 with discriminative learning rates
- 📊 **Comprehensive EDA**: Detailed exploratory data analysis
- 🌐 **Interactive Web App**: Streamlit-based deployment with batch processing
- 📈 **Production Ready**: Complete pipeline from data to deployment

---

## ✨ Features

### Core Functionality
- **Binary Classification**: Parasitized vs. Uninfected blood cells
- **Transfer Learning**: Pre-trained ResNet18 fine-tuned on malaria dataset
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Discriminative Learning Rates**: Layer-wise learning rate optimization
- **Early Stopping**: Automatic model checkpointing based on validation accuracy

### Visualization & Interpretability
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Training Curves**: Loss and accuracy plots
- **Confusion Matrix**: Detailed performance breakdown
- **Sample Predictions**: Visual inspection of model outputs

### Web Application
- **Single Image Analysis**: Detailed prediction with Grad-CAM
- **Batch Processing**: Analyze multiple images simultaneously
- **Confidence Thresholding**: Flag uncertain predictions
- **Prediction History**: Track and export results
- **Interactive UI**: User-friendly Streamlit interface

---

## 📊 Dataset

### Source
The dataset consists of microscopic blood cell images divided into two classes:
- **Parasitized**: Cells infected with malaria parasites
- **Uninfected**: Healthy blood cells

### Dataset Statistics
```
Total Images: ~27,558
├── Training Set: 70% (~19,290 images)
├── Validation Set: 15% (~4,134 images)
└── Test Set: 15% (~4,134 images)

Class Distribution:
├── Parasitized: ~50%
└── Uninfected: ~50%
```

### Image Properties
- **Format**: PNG/JPG
- **Original Size**: Variable (typically 130x130 pixels)
- **Input Size**: Resized to 224x224 for ResNet18
- **Color**: RGB (3 channels)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/malaria-detection.git
cd malaria-detection
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n malaria python=3.8
conda activate malaria
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
```bash
# Option 1: Manual download
# Download from [source] and place in ./data/

# Option 2: Using script
python scripts/download_data.py
```

---

## 📁 Project Structure

```
malaria-detection/
│
├── data/
│   ├── raw/                          # Original dataset
│   │   ├── Parasitized/
│   │   └── Uninfected/
│   └── split/                        # Train/Val/Test split
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA notebook
│   ├── 02_model_training.ipynb       # Training experiments
│   └── 03_evaluation.ipynb           # Model evaluation
│
├── scripts/
│   ├── split_dataset.py              # Data splitting script
│   ├── eda.py                        # Exploratory data analysis
│   ├── train_resnet18.py             # Training script
│   └── evaluate.py                   # Evaluation script
│
├── models/
│   └── resnet18_malaria_best.pth     # Trained model weights
│
├── results/
│   ├── training_history.png          # Training curves
│   ├── confusion_matrix.png          # Confusion matrix
│   ├── class_distribution.png        # Class distribution
│   ├── dimensions_analysis.png       # Image size analysis
│   ├── sample_images.png             # Sample images
│   └── intensity_analysis.png        # Pixel intensity analysis
│
├── app/
│   └── app_gradcam.py                # Streamlit web application
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── LICENSE                            # License information
```

---

## 💻 Usage

### 1. Data Preparation

#### Split Dataset
```bash
python scripts/split_dataset.py
```

This will create train/val/test splits with 70/15/15 ratio.

#### Exploratory Data Analysis
```bash
python scripts/eda.py
```

Generates comprehensive visualizations of the dataset.

### 2. Model Training

#### Basic Training
```bash
python scripts/train_resnet18.py
```

#### Custom Training Parameters
```python
# Edit train_resnet18.py
num_epochs = 20
batch_size = 32
learning_rate = 0.001
```

#### Training with Fine-tuning
The training script implements:
- Layer unfreezing (layer3, layer4, fc)
- Discriminative learning rates:
  - `fc`: 0.001
  - `layer4`: 0.0001
  - `layer3`: 0.00001
- Learning rate scheduling
- Model checkpointing

### 3. Model Evaluation

```bash
python scripts/evaluate.py --model models/resnet18_malaria_best.pth
```

### 4. Web Application

#### Launch Streamlit App
```bash
streamlit run app/app_gradcam.py
```

The app will be available at `http://localhost:8501`

#### App Features
- **Single Image Mode**: Upload and analyze one image
- **Batch Mode**: Process multiple images simultaneously
- **Grad-CAM Visualization**: See model attention
- **Download Results**: Export predictions and visualizations
- **Prediction History**: Track all analyses

---

## 📈 Model Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.22% |
| **Validation Accuracy** | 97.05% |
| **Training Time** | ~15-20 minutes (GPU) |
| **Model Size** | ~44 MB |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **Parasitized** | 0.97 | 0.96 | 0.97 | 96.23% |
| **Uninfected** | 0.97 | 0.98 | 0.97 | 98.21% |

### Confusion Matrix

```
                Predicted
              P         U
Actual  P  [1982]    [78]
        U   [37]   [2037]
```

### Training History

- **Best Epoch**: 9/10
- **Final Training Loss**: 0.0812
- **Final Validation Loss**: 0.0891
- **Convergence**: Achieved by epoch 7

---

## 🌐 Web Application

### Features

#### Single Image Analysis
1. Upload blood cell image
2. Receive prediction with confidence score
3. View Grad-CAM heatmap showing model attention
4. Download results

#### Batch Processing
1. Upload multiple images (supports 10-100+ images)
2. Automatic processing with progress tracking
3. Summary table with all predictions
4. Flag low-confidence predictions for review
5. Export results as CSV

#### Settings
- **Confidence Threshold**: Adjustable (default: 85%)
- **Grad-CAM Opacity**: Customizable overlay transparency
- **Batch Mode Toggle**: Switch between single/batch processing

#### Prediction History
- Session-based tracking
- Statistics dashboard
- Export capability
- Clear history option

### Screenshots

[Add screenshots of your app here]

---

## 🔬 Results

### Key Findings

1. **Discriminative Learning Rates**: Effective strategy achieving 97.22% accuracy
2. **Balanced Performance**: Both classes achieve >96% accuracy
3. **Generalization**: Validation and test accuracies closely aligned (minimal overfitting)
4. **Interpretability**: Grad-CAM shows model focuses on parasite regions

### Visualizations

All results visualizations are saved in the `results/` directory:
- Training/validation curves
- Confusion matrix
- Sample predictions with Grad-CAM
- Class distribution analysis
- Image dimension analysis

---

## 🚧 Future Improvements

### Short-term
- [ ] Implement Test-Time Augmentation (TTA)
- [ ] Add model ensemble (ResNet34, ResNet50)
- [ ] Optimize model for mobile deployment
- [ ] Add more data augmentation techniques

### Medium-term
- [ ] Multi-class classification (parasite species)
- [ ] Implement attention mechanisms
- [ ] Add explainability techniques (SHAP, LIME)
- [ ] Create REST API for integration

### Long-term
- [ ] Develop mobile application
- [ ] Real-time video analysis
- [ ] Integration with laboratory systems
- [ ] Clinical validation studies

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/publication/pub9932)
- **Model Architecture**: ResNet18 from torchvision
- **Framework**: PyTorch, Streamlit
- **Inspiration**: Medical AI research community

### References

1. Rajaraman, S., et al. (2018). "Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images." PeerJ.
2. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

---

## 📧 Contact

**Project Maintainer**: Your Name
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## ⚠️ Medical Disclaimer

**This tool is for educational and research purposes only.** It is not intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical diagnosis and treatment. The developers assume no liability for any medical decisions made based on the outputs of this system.

---

## 📊 Project Status

![Status](https://img.shields.io/badge/status-active-success.svg)
![Maintenance](https://img.shields.io/badge/maintenance-yes-green.svg)
![Last Commit](https://img.shields.io/github/last-commit/yourusername/malaria-detection)

**Current Version**: 1.0.0  
**Last Updated**: January 2026

---

<div align="center">

### 🌟 If you find this project helpful, please consider giving it a star! 🌟

Made with ❤️ by [Your Name]

</div>