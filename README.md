# PathMNIST Colorectal Cancer Tissue Classification

A deep learning project for multi-class classification of colorectal cancer histology patches using the [PathMNIST](https://medmnist.com/) benchmark dataset, with Grad-CAM visualizations for model interpretability.

---

## Project Overview

This project trains and evaluates CNN-based classifiers (ResNet18 and VGG16, both from-scratch and pretrained) to classify 28×28 RGB histopathology images into 9 tissue categories. A custom Grad-CAM implementation is provided to produce saliency maps that highlight the image regions most influential to each prediction.

**Dataset:** PathMNIST (subset of MedMNIST v2)
- **Task:** Multi-class classification (9 classes)
- **Image size:** 28×28, RGB (3 channels)
- **Train size:** 89,996 | **Validation size:** 10,004
- **Classes:** adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium

---

## File Structure

```
.
├── main.ipynb          # Main notebook: EDA, training, evaluation, Grad-CAM
├── train.py            # Training loop, TrainConfig, and TrainHistoryRecords
├── gradcam_utils.py    # Grad-CAM implementation and visualization helpers
├── checkpoints/        # Saved model checkpoints (created at runtime)
├── meta_logs/          # Training history CSVs (created at runtime)
└── README.md
```

---

## Models

Four model configurations were trained and compared:

| Parameter | ResNet (scratch) | ResNet (pretrained) | VGG16 | ResNet (scratch) | ResNet (pretrained) | VGG16 |
|-----------|------------------|---------------------|-------|------------------|---------------------|-------|
| **Optimizer** | Adam | Adam | Adam | SGD | SGD | SGD |
| **Epochs** | 20 | 20 | 20 | 20 | 20 | 20 |
| **Learning rate** | 0.001 | 0.001 | 0.001 | 0.003 | 0.003 | 0.003 |
| **Batch size** | 64 | 64 | 64 | 128 | 128 | 128 |
| **Loss** | 0.0487 | 0.3716 | 0.8337 | 0.0792 | 0.3806 | 0.8220 |
| **Accuracy** | 0.9856 | 0.8679 | 0.7296 | 0.9749 | 0.8780 | 0.7345 |
| **AUC** | 0.9997 | 0.9881 | 0.9567 | 0.9993 | 0.9888 | 0.9568 |
| **Macro-F1** | 0.9855 | 0.8672 | 0.7283 | 0.9748 | 0.8778 | 0.7331 |

ResNet18 trained from scratch outperformed pretrained variants on this dataset, likely due to the domain gap between ImageNet and histopathology images.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook main.ipynb
```

The dataset will be downloaded automatically via the `medmnist` package on first run.

---

## Training

Training is configured via `TrainConfig` in `train.py`:

```python
from train import train, TrainConfig

config = TrainConfig(
    epochs=20,
    lr=1e-3,
    batch_size=64,
    optimizer_cls=torch.optim.Adam,
    device="cuda",
    save_checkpoints=True,
    early_stopping_patience=5,
)
history = train(model=model, config=config)
```

**Key `TrainConfig` options:**

| Parameter               | Default       | Description                                      |
|------------------------|---------------|--------------------------------------------------|
| `epochs`               | 10            | Number of training epochs                        |
| `lr`                   | 1e-4          | Learning rate                                    |
| `batch_size`           | 64            | Batch size                                       |
| `optimizer_cls`        | `Adam`        | Optimizer class                                  |
| `device`               | auto (cuda/cpu) | Training device                                |
| `save_checkpoints`     | False         | Whether to save `.pt` checkpoints                |
| `checkpoints_dir`      | `checkpoints` | Directory for checkpoint files                   |
| `save_frequency`       | 2             | Save a checkpoint every N epochs                 |
| `save_best_only`       | True          | Also save `best.pt` whenever validation improves |
| `early_stopping_patience` | —          | Stop early if validation does not improve        |

Checkpoints are saved to `checkpoints/<run_name>/` and training metrics are logged as CSV files under `meta_logs/`.

---

## Grad-CAM Visualization

`gradcam_utils.py` provides a minimal, hook-based Grad-CAM implementation that works with any CNN backbone.

```python
from gradcam_utils import GradCAM, replace_relu_inplace, show_gradcam_result

# Replace in-place ReLUs to allow gradient flow
replace_relu_inplace(model)
model.eval()

# Attach Grad-CAM to a target layer (dotted path or module object)
gradcam = GradCAM(model, target_layer="layer4.1.conv2")

# Run on a single image tensor [1, C, H, W]
saliency_map, logits = gradcam(input_tensor)

# Visualize
show_gradcam_result(
    image_tensor=input_tensor,
    mask=saliency_map,
    mean=train_mean,
    std=train_sd,
    label_map=label_map,
    true_label=true_label,
    pred_label=logits.argmax(dim=1).item(),
)

gradcam.remove_hooks()
```

For batch visualization, use `show_gradcam_grid()`.

---

## Evaluation Metrics

- **Accuracy** — overall classification accuracy
- **AUC** — macro-averaged one-vs-rest ROC AUC
- **Macro-F1** — unweighted mean F1 across all 9 classes

---

## Requirements

See `requirements.txt`. Key dependencies: PyTorch, torchvision, medmnist, scikit-learn, matplotlib, numpy, pandas.

---

## Notes

- Images are 28×28; resizing to 224×224 is optional and commented out in the notebook.
- Dataset mean and standard deviation are computed from the training split at runtime.
- All in-place ReLU layers must be replaced before running Grad-CAM (use `replace_relu_inplace`).
- Python 3.11 was used during development.
