#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # STEP 1: Install Kaggle API client
# !pip install -q kaggle

# # STEP 2: Upload your Kaggle API key (kaggle.json)
# from google.colab import files
# files.upload()  # upload kaggle.json manually when prompted

# # STEP 3: Setup Kaggle credentials
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# # STEP 4: Download the dataset
# !kaggle datasets download -d kabil007/lungcancer4types-imagedataset

# # STEP 5: Unzip the dataset
# !unzip -q lungcancer4types-imagedataset.zip -d lung_cancer_dataset


# In[2]:


get_ipython().system('pip install torchinfo')
get_ipython().system('pip install --quiet torchview graphviz')


# In[3]:


import os
import time
import math
from datetime import timedelta

import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from timm.models.vision_transformer import VisionTransformer
from torchinfo import summary
from torchview import draw_graph
from IPython.display import SVG, display


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


# Configuration
data_dir = '/kaggle/input/Data'
# data_dir = '/content/lung_cancer_dataset/Data'
batch_size = 8
image_size = 224
num_workers = 4
pin_memory = torch.cuda.is_available()

basic_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

remap_classes = {
    'adenocarcinoma': 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma': 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'squamous.cell.carcinoma': 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa',
}


# In[6]:


class RemapImageFolder(datasets.ImageFolder):
    def __init__(self, root, remap_dict=None, transform=None):
        super().__init__(root, transform=transform)
        self.remap_dict = remap_dict or {}

        # Build new classes set after remapping
        new_classes = set()
        for cls in self.classes:
            new_classes.add(self.remap_dict.get(cls, cls))
        self.classes = sorted(new_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Remap samples to new class indices
        new_samples = []
        for path, _ in self.samples:
            folder_name = os.path.basename(os.path.dirname(path))
            mapped_class = self.remap_dict.get(folder_name, folder_name)
            new_samples.append((path, self.class_to_idx[mapped_class]))
        self.samples = new_samples

        # Update targets accordingly
        self.targets = [s[1] for s in self.samples]


# Load datasets with remapping for train, valid, and test
train_dataset = RemapImageFolder(root=os.path.join(data_dir, 'train'), remap_dict=remap_classes, transform=basic_transform)
valid_dataset = RemapImageFolder(root=os.path.join(data_dir, 'valid'), remap_dict=remap_classes, transform=basic_transform)
test_dataset = RemapImageFolder(root=os.path.join(data_dir, 'test'), remap_dict=remap_classes, transform=basic_transform)

# Create separate DataLoaders for each split
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory_device='cuda' if torch.cuda.is_available() else ''
)

val_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory_device='cuda' if torch.cuda.is_available() else ''
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory_device='cuda' if torch.cuda.is_available() else ''
)


# In[7]:


from collections import Counter

def display_split_stats(train_dataset, valid_dataset, test_dataset):
    # Total number of images in each split
    train_count = len(train_dataset)
    valid_count = len(valid_dataset)
    test_count = len(test_dataset)
    total = train_count + valid_count + test_count

    print(f"Total images: {total}")
    print(f"Train images: {train_count} ({train_count/total*100:.2f}%)")
    print(f"Valid images: {valid_count} ({valid_count/total*100:.2f}%)")
    print(f"Test images:  {test_count} ({test_count/total*100:.2f}%)")
    print("\n")

    # Helper to count images per class
    def count_per_class(dataset):
        counter = Counter()
        for target in dataset.targets:
            counter[target] += 1
        return counter

    train_counts = count_per_class(train_dataset)
    valid_counts = count_per_class(valid_dataset)
    test_counts = count_per_class(test_dataset)

    print(f"{'Class':50s} | {'Train':>5s} | {'Valid':>5s} | {'Test':>5s}")
    print("-" * 75)
    for idx, cls_name in enumerate(train_dataset.classes):
        tr = train_counts.get(idx, 0)
        va = valid_counts.get(idx, 0)
        te = test_counts.get(idx, 0)
        print(f"{cls_name:50s} | {tr:5d} | {va:5d} | {te:5d}")

# Usage
display_split_stats(train_dataset, valid_dataset, test_dataset)


# In[8]:


def plot_random_images(dataset, rows=3, cols=3):
    num_images = rows * cols
    plt.figure(figsize=(cols * 2.5, rows * 2.5))

    indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        img_np = image.permute(1, 2, 0).numpy()

        full_class_name = dataset.classes[label]
        clipped_name = full_class_name.split('.')[0]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_np)
        plt.title(clipped_name, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_random_images(train_dataset, rows=3, cols=3)


# In[9]:


class MaxAvgPool2d(nn.Module):
    def __init__(self, pool_size=(2, 2), stride=None, padding="same"):
        """
        Concatenates MaxPooling and AvgPooling outputs along the channel dimension.
        Always does manual 'same'-style padding (or explicit int/tuple padding) to avoid
        any 'padding="same"' argument.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride or pool_size

        # Determine padding amounts
        if isinstance(padding, str) and padding.lower() == "same":
            # “same” padding: pad so output H_out = ceil(H_in / stride)
            # For each dimension: pad_total = max((ceil(H_in/stride) - 1)*stride + k - H_in, 0)
            # But a simple symmetric version is floor((k-1)/2) on each side.
            kH, kW = pool_size
            self.pad_h = (kH - 1) // 2
            self.pad_w = (kW - 1) // 2
        else:
            # If padding is an int or tuple
            if isinstance(padding, int):
                self.pad_h = self.pad_w = padding
            else:
                # assume padding is a 2‐tuple
                self.pad_h, self.pad_w = padding

        # Create pooling layers (no built-in padding here; we'll pad manually)
        self.max_pool = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, 2*C, H_out, W_out)
        """
        # 1) Manually pad
        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        x_padded = F.pad(
            x,
            (self.pad_w, self.pad_w, self.pad_h, self.pad_h),
            mode="constant",
            value=0
        )

        # 2) Perform max- and avg-pooling (padding=0 because we've already padded)
        max_out = self.max_pool(x_padded)
        avg_out = self.avg_pool(x_padded)

        # 3) Concatenate along the channel dimension
        out = torch.cat([max_out, avg_out], dim=1)
        return out


# In[10]:


class CNN_ViT_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 1) Load MobileNetV2
        weights = MobileNet_V2_Weights.DEFAULT
        base = mobilenet_v2(weights=weights)
        total_blocks = len(base.features)

        # Freeze first half
        for i, blk in enumerate(base.features):
            if i < total_blocks // 2:
                for p in blk.parameters():
                    p.requires_grad = False

        # 2) Split into: pre-4th-last, 4th-last, and rest
        self.stage_pre4  = nn.Sequential(*base.features[: total_blocks - 2])
        self.stage4      = base.features[total_blocks - 2]
        self.stage_rest  = nn.Sequential(*base.features[total_blocks - 1:])  # last 3 blocks

        # Channel counts
        ch_stage4 = self.stage4.out_channels     # e.g. 160
        ch_final  = base.features[-1].out_channels  # 1280

        # 3) CNN→ViT head
        self.cnn_to_vit = nn.Sequential(
            nn.Conv2d(ch_final, 256, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

        # 4) Vision Transformer (feature extractor only)
        embed_dim = 64
        self.vit = VisionTransformer(
            img_size=28, patch_size=7, in_chans=32,
            num_classes=0, embed_dim=embed_dim,
            depth=2, num_heads=4, mlp_ratio=2,
            qkv_bias=True, norm_layer=nn.LayerNorm,
        )
        self.vit.head = nn.Identity()

        # 5) Parallel on final features
        self.parallel1 = nn.Sequential(
            nn.Conv2d(ch_final, 128, 1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

        # 7) Classifier: [vit + p1 + p2]
        self.classifier = nn.Linear(embed_dim + 64, num_classes)

    def forward(self, x):
        # 4th-last output
        x_pre4   = self.stage_pre4(x)      # [B, C_pre4, H4, W4]
        feature4 = self.stage4(x_pre4)     # [B, ch_stage4, H3, W3]

        # Final backbone output
        feat1    = self.stage_rest(feature4)  # [B, ch_final, H2, W2]

        # ViT branch
        vit_in   = self.cnn_to_vit(feat1)     # [B, 32, 28, 28]
        vit_out  = self.vit(vit_in)           # [B, embed_dim]

        # Parallel branches
        p1       = self.parallel1(feat1)      # [B, 64]

        # Concat & classify
        combined = torch.cat([vit_out, p1], dim=1)  # [B, embed_dim+128]
        return self.classifier(combined)


# In[11]:


model = CNN_ViT_Model(num_classes=4)
summary(model, input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"])


# In[12]:


model.to(device)

# Dummy input (can adjust shape based on your model's expected input)
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# Generate the graph — expand_nested=True if model has nested submodules
model_graph = draw_graph(
    model,
    input_data=dummy_input,
    expand_nested=True,
    graph_name="CNN_ViT_Model",
    save_graph=False  # We don't want to save, only display inline
)

# Render to SVG and display inline in the notebook
svg_output = model_graph.visual_graph.pipe(format='svg')
display(SVG(svg_output))


# In[13]:


def custom_metrics(y_pred, y_true, loss):
    y_pred_classes = torch.argmax(y_pred, dim=1)

    y_true_numpy = y_true.cpu().numpy()
    y_pred_classes_numpy = y_pred_classes.cpu().numpy()
    accuracy = accuracy_score(y_true_numpy, y_pred_classes_numpy)

    precision = precision_score(y_true_numpy, y_pred_classes_numpy, average='weighted', zero_division=0)
    recall = recall_score(y_true_numpy, y_pred_classes_numpy, average='weighted', zero_division=0)
    f1 = f1_score(y_true_numpy, y_pred_classes_numpy, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true_numpy, y_pred_classes_numpy)

    tn = cm[0, 0]
    fp = cm[0, 1:].sum()
    fn = cm[1:, 0].sum()
    tp = cm[1:, 1:].sum()

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0

    mcc = matthews_corrcoef(y_true_numpy, y_pred_classes_numpy)

    num_classes = y_pred.shape[1]
    auc_scores = []

    for class_idx in range(num_classes):
        class_y_true = (y_true_numpy == class_idx).astype(np.float32)
        class_y_pred = y_pred[:, class_idx].cpu().numpy()

        # Skip AUC if only one class is present
        if np.unique(class_y_true).size < 2:
            continue
        try:
            auc_score = roc_auc_score(class_y_true, class_y_pred)
            auc_scores.append(auc_score)
        except ValueError:
            continue

    auc_avg = np.mean(auc_scores) if auc_scores else 0.0

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "mcc": mcc,
        "auc": auc_avg,
    }

    return metrics


# In[14]:


# Hyperparameters
num_epochs       = 35
initial_lr       = 1e-4
decay_start_epoch = 10
decay_factor     = 0.97

# Criterion, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

# Scheduler policy: 1.0 until epoch 11, then 0.98^(epoch - 11)
lr_lambda = lambda epoch: 1.0 if epoch < decay_start_epoch else decay_factor ** (epoch - decay_start_epoch)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# In[15]:


# Metrics initialization
metric_names = ["loss", "accuracy", "precision", "recall", "f1_score",
                "specificity", "sensitivity", "mcc", "auc"]
train_metrics_history = {metric: [] for metric in metric_names}
val_metrics_history = {metric: [] for metric in metric_names}

total_training_time = 0.0
total_validation_time = 0.0

# Training and validation loop
for epoch in range(num_epochs):
    start_time = time.time()

    # Training loop
    model.train()
    total_train_loss = 0.0
    all_train_predictions = []
    all_train_targets = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch [{epoch + 1}/{num_epochs}]")):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        all_train_predictions.extend(outputs.detach().cpu().numpy())
        all_train_targets.extend(targets.detach().cpu().numpy())

    end_time = time.time()
    epoch_training_time = end_time - start_time
    total_training_time += epoch_training_time

    average_train_loss = total_train_loss / len(train_loader)
    all_train_predictions = np.array(all_train_predictions)
    all_train_targets = np.array(all_train_targets)

    train_metrics = custom_metrics(torch.tensor(all_train_predictions), torch.tensor(all_train_targets), average_train_loss)
    training_metrics_line = "Train Metrics - " + ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
    print(training_metrics_line)

    for metric in metric_names:
        train_metrics_history[metric].append(train_metrics[metric])

    # Validation loop
    model.eval()
    total_val_loss = 0.0
    all_val_predictions = []
    all_val_targets = []
    start_val_time = time.time()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(val_loader, desc="Validating Model")):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

            all_val_predictions.extend(outputs.detach().cpu().numpy())
            all_val_targets.extend(targets.detach().cpu().numpy())

    end_val_time = time.time()
    epoch_validation_time = end_val_time - start_val_time
    total_validation_time += epoch_validation_time

    average_val_loss = total_val_loss / len(val_loader)
    all_val_predictions = np.array(all_val_predictions)
    all_val_targets = np.array(all_val_targets)

    val_metrics = custom_metrics(torch.tensor(all_val_predictions), torch.tensor(all_val_targets), average_val_loss)
    validation_metrics_line = "Val Metrics - " + ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
    print(validation_metrics_line)

    for metric in metric_names:
        val_metrics_history[metric].append(val_metrics[metric])

    # Update LR via scheduler rather than manual mutliplication
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}/{num_epochs} — Loss: {loss.item():.4f} — LR: {current_lr:.6f}")

print(f"Total Training Time: {str(timedelta(seconds=total_training_time))}")
print(f"Total Validation Time: {str(timedelta(seconds=total_validation_time))}")

avg_training_time_per_epoch = total_training_time / (epoch + 1)
avg_validation_time_per_epoch = total_validation_time / (epoch + 1)
print(f"Average Training Time per Epoch: {str(timedelta(seconds=avg_training_time_per_epoch))}")
print(f"Average Validation Time per Epoch: {str(timedelta(seconds=avg_validation_time_per_epoch))}")


# In[16]:


def plot_metrics(train_metrics_history, val_metrics_history, metric_names):
    num_metrics = len(metric_names)
    num_epochs = len(train_metrics_history[metric_names[0]])

    figure, axes = plt.subplots(num_metrics, figsize=(10, 6 * num_metrics))
    rng = range(1, num_epochs + 1)

    for ax, metric_name in zip(axes, metric_names):
        train_metric = train_metrics_history[metric_name]
        val_metric = val_metrics_history[metric_name]

        ax.plot(rng, train_metric, label="Training")
        ax.plot(rng, val_metric, label="Validation")
        ax.legend()
        ax.set_xlabel("Epochs")

        if metric_name in ("auc", "mcc"):
            ax.set_ylabel(metric_name.upper())
            ax.set_title(f"{metric_name.upper()} vs Epochs")
        else:
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f"{metric_name.capitalize()} vs Epochs")

        max_metric = max(max(train_metric), max(val_metric))
        min_metric = min(min(train_metric), min(val_metric))
        y_max = math.ceil(max_metric)

        if min_metric > 0 or max_metric > 1:
            ax.set_ylim(0, y_max)
        else:
            ax.set_ylim(min_metric, y_max)

        ax.grid(True, linestyle='--', alpha=0.5)
        # Adjust xlim to avoid identical low and high limits.
        if num_epochs == 1:
            ax.set_xlim(0.5, 1.5)
        else:
            ax.set_xlim(1, num_epochs)

    plt.tight_layout()
    plt.show()

plot_metrics(train_metrics_history, val_metrics_history, metric_names)


# In[17]:


# Testing loop
model.eval()
total_test_loss = 0.0
all_test_predictions = []
all_test_targets = []

with torch.no_grad():
    for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Testing Model")):
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)
        total_test_loss += loss.item()

        all_test_predictions.extend(outputs.detach().cpu().numpy())
        all_test_targets.extend(targets.detach().cpu().numpy())

all_test_predictions = np.array(all_test_predictions)
all_test_targets = np.array(all_test_targets)

average_test_loss = total_test_loss / len(test_loader)
test_metrics = custom_metrics(torch.tensor(all_test_predictions), torch.tensor(all_test_targets), average_test_loss)
testing_metrics_line = "Test Metrics - " + ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
print(testing_metrics_line)


# In[18]:


# Convert true_labels and predicted_labels to numpy arrays
true_labels = np.array(all_test_targets)

# Assuming predicted_labels are probabilities, convert them to class labels
predicted_labels = np.argmax(np.array(all_test_predictions), axis=1)

# Ensure both true_labels and predicted_labels are of integer type
true_labels = true_labels.astype(int)
predicted_labels = predicted_labels.astype(int)

# # Option 2: if using ImageFolder
# classes = dataset.classes

# Get class names from the dataset
classes = train_dataset.classes

# classes = label_encoder.classes_

# Generate a classification report
report = classification_report(true_labels, predicted_labels, target_names=classes, digits=4)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
num_errors = np.sum(true_labels != predicted_labels)

print(report)
print(f'There were {num_errors} errors in {len(predicted_labels)} tests for an accuracy of {accuracy*100:6.2f}')

