# 🏙️ Crowd Counting with CSRNet - Festival Harmoni Nusantara

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

*AI-powered crowd monitoring system untuk keamanan dan efisiensi kota cerdas*

[Demo](#demo) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## 📖 Overview

<div align="center">
  <img src="https://github.com/user-attachments/assets/bbb061b0-1b8a-4ea7-8040-a500ad0e517b" alt="Crowd Counting CSRNet" width="80%">
</div>

<br>

Sistem crowd counting berbasis deep learning untuk membantu petugas perencanaan kota dalam memantau kepadatan pengunjung pada Festival Harmoni Nusantara. Model ini menggunakan **Improved CSRNet** yang dapat menghitung jumlah orang secara akurat dari gambar CCTV, membantu mencegah overcrowding, mengoptimalkan jalur evakuasi, dan memastikan pengalaman pengunjung tetap aman.


### ✨ Key Features

- **Accurate Counting**: Prediksi jumlah kerumunan dengan density map estimation yang akurat
- **Multi-Scale Architecture**: Backend multi-branch untuk menangkap informasi dari berbagai skala
- **Robust Augmentation**: Multi-scale dan multi-crop augmentation untuk data diversity
- **Real-time Monitoring**: Dapat diintegrasikan dengan sistem CCTV untuk monitoring real-time
- **Adaptive Density Map**: Gaussian kernel adaptif berdasarkan kepadatan lokal

---

## 🎯 Problem Statement

Abi, petugas perencanaan kota di Badan Pengelola Kota Cerdas Nusantara, menghadapi tantangan kompleks dalam mengantisipasi ratusan ribu pengunjung Festival Harmoni Nusantara. Dibutuhkan sistem AI yang dapat[file:1]:

- Memantau potensi overcrowding secara real-time
- Mengidentifikasi tingkat kepadatan di berbagai area
- Mengoptimalkan jalur evakuasi berdasarkan distribusi kerumunan
- Memastikan keamanan dan kenyamanan pengunjung

---

## 🏗️ Architecture

### Model Overview

Model ini menggunakan **Improved CSRNet** yang terdiri dari dua komponen utama[file:1][web:3][web:7]:

**Frontend (VGG-16)**
- 10 layer pertama VGG-16 pre-trained pada ImageNet
- 3x max pooling untuk ekstraksi fitur hierarkis
- Output: Feature map 1/8 resolusi input

**Backend (Multi-Branch)**
```

Branch 1: Dilated Conv (dilation=2) → 256 channels
Branch 2: Bottleneck (1x1→3x3→1x1) → 256 channels
Branch 3: Bottleneck (1x1→5x5→1x1) → 256 channels
Branch 4: Conv (1x1) → 128 channels
─────────────────────────────────────────────────
Concatenation → 896 channels
Fusion Layers → 64 channels
Output Layer → 1 channel (Density Map)

```


### Loss Function

Combined loss untuk optimasi ganda:
- **MSE Loss**: Kualitas density map
- **L1 Loss (MAE)**: Akurasi counting (weight: 0.01)

---

## 🚀 Installation

### Requirements

```

Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.0 (recommended)

```

### Setup

```


# Clone repository

git clone https://github.com/Fauzan-A25/crowd-counting-with-csrnet-competition-hology.git
cd crowd-counting-with-csrnet-competition-hology

# Install dependencies

pip install torch torchvision numpy opencv-python scipy pillow tqdm

```

---

## 📊 Dataset Structure

```

dataset/
├── train/
│   ├── images/          \# Training images
│   └── labels/          \# JSON annotation files
├── test/
│   └── images/          \# Test images

```

Setiap file JSON berisi array koordinat titik kerumunan:
```

{
"points": [[x1, y1], [x2, y2], ...]
}

```

---

## 💻 Usage

### Training

```


# Set parameters

train_img_dir = "dataset/train/images"
train_json_dir = "dataset/train/labels"
epochs = 50
batch_size = 4
learning_rate = 1e-4

# Initialize dataset and dataloader

train_dataset = CrowdDataset(train_img_dir, train_json_dir,
target_size=(256,256),
use_crop=True,
multi_scale=True)

# Initialize model

model = ImprovedCSRNet()

# Train

train_model(model, train_loader, val_loader, epochs=epochs)

```

### Inference

```


# Load trained model

model = ImprovedCSRNet()
model.load_state_dict(torch.load('best_csrnet.pth'))
model.eval()

# Predict

test_dir = "dataset/test/images"
predictions = predict_on_test(model, test_dir)

# Save results

predictions.to_csv('submission.csv', index=False)

```

### Resume Training

```


# Continue from checkpoint

train_model(model, train_loader, val_loader,
epochs=100,
start_epoch=50,
checkpoint_path='checkpoint_epoch_50.pth')

```

---

## 🔬 Technical Details

### Data Preprocessing

1. **Multi-Scale Augmentation**: Random scaling (0.7-1.3x) untuk variasi ukuran
2. **Canvas Resizing**: Resize ke 640x360 (16:9 ratio) dengan padding
3. **Density Map Generation**: Adaptive Gaussian kernel dengan KDTree untuk neighbor detection
4. **Multi-Crop**: Split gambar menjadi 4 atau 9 patches untuk augmentasi
5. **Normalization**: ImageNet mean/std normalization

### Training Strategy

- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Mixed Precision**: torch.cuda.amp untuk efisiensi GPU
- **Validation Split**: 80% train, 20% validation
- **Checkpointing**: Save setiap 2 epoch + best model berdasarkan MAE

---

## 📈 Results

Model ini menggunakan metrik evaluasi:
- **MAE (Mean Absolute Error)**: Rata-rata selisih prediksi dengan ground truth
- **MSE (Mean Squared Error)**: Kualitas density map

Training progress divisualisasikan dengan tqdm progress bar menampilkan loss, prediksi, dan MAE per batch.

---

## 🛠️ Model Checkpoints

```

best_csrnet.pth           \# Model dengan validation MAE terendah
checkpoint_epoch_X.pth    \# Checkpoint setiap 2 epoch

```

---

## 📚 Documentation

### Key Classes

**CrowdDataset**
- Custom PyTorch Dataset untuk loading images dan annotations
- Multi-scale dan multi-crop augmentation
- Adaptive density map generation

**ImprovedCSRNet**
- VGG-16 frontend untuk feature extraction
- Multi-branch backend untuk multi-scale context
- Dilated convolutions untuk receptive field expansion

---

## 🎓 References

1. Li, Y., Zhang, X., & Chen, D. (2018). CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes. CVPR 2018
2. Improved CSRNet Method, ICAROB 2020
3. Adaptive Density Map Generation for Crowd Counting, ICCV 2019

---

## 🤝 Contributing

Contributions are welcome! Silakan buat pull request atau open issue untuk:
- Bug fixes
- Feature improvements
- Documentation enhancements
- Dataset expansion

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Fauzan Ardiansyah**

- GitHub: [@Fauzan-A25](https://github.com/Fauzan-A25)
- Project: [crowd-counting-with-csrnet-competition-hology](https://github.com/Fauzan-A25/crowd-counting-with-csrnet-competition-hology)

---

## 🙏 Acknowledgments

- Hology Competition 8.0 untuk challenge yang menarik
- VGG Team untuk pre-trained weights
- PyTorch Community untuk framework yang powerful[file:1]

---

<div align="center">

**⭐ Star this repo jika bermanfaat! ⭐**

</div>
