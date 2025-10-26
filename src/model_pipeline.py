"""
model_pipeline.py

한 파일에 Dataset, Model, Training, Evaluation을 포함한 파이프라인입니다.
4번(이미지 생성)까지 완료되어 images/와 labels.csv가 준비되었다는 전제입니다.

사용법 예시:
  # 학습
  python src/model_pipeline.py --mode train --data_csv data/processed/labels.csv --images_dir data/images --epochs 50 --batch_size 32

  # 평가
  python src/model_pipeline.py --mode eval --data_csv data/processed/labels.csv --images_dir data/images --checkpoint models/best_model.pth

필요 패키지:
  torch torchvision pandas numpy scikit-learn pillow pyyaml tqdm

구성:
  - ChartDataset: PyTorch Dataset (이미지 로드, transform)
  - get_dataloaders: train/val/test 분할
  - build_model: ResNet18 기반 커스텀 헤드
  - FocalLoss: Class imbalance를 효과적으로 다루는 loss function
  - train: Focal Loss를 사용한 학습 루프
  - evaluate: AUC, accuracy, precision, recall, confusion matrix

주의사항:
  - GPU 사용 시 CUDA 가용성을 확인하세요.
  - Focal Loss의 alpha는 클래스 비율에 따라 자동 계산됩니다.
"""

import os
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs: Predicted probabilities (after sigmoid), shape (N,)
            targets: Ground truth labels (0 or 1), shape (N,)
        """
        eps = 1e-7
        outputs = outputs.clamp(eps, 1 - eps)

        # p_t: probability of the true class
        p_t = outputs * targets + (1 - outputs) * (1 - targets)

        # alpha_t: class-dependent weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # focal loss
        focal_weight = (1 - p_t) ** self.gamma
        loss = -alpha_t * focal_weight * torch.log(p_t)

        return loss.mean()


class ChartDataset(Dataset):
    """이미지 경로와 라벨이 담긴 CSV로부터 Dataset을 만듭니다.
    CSV는 최소한 columns: image_path, label (0/1) 을 포함해야 합니다.
    image_path는 images_dir 기준 상대경로이거나 절대경로일 수 있습니다
    """

    def __init__(self, df: pd.DataFrame, images_dir: str = '.', image_col: str = 'image_path', label_col: str = 'label',
                 image_size: int = 224, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = image_size
        self.train = train

        # transforms
        if self.train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.2),
                T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        p = Path(img_path)

        # --- Fix for double-prefix issue ---------------------------------
        # Avoid doubling images_dir when image_path already contains it
        # e.g. if image_path == 'data/images/KRW_BTC_...png' and images_dir == 'data/images'
        # previously: images_dir / image_path -> 'data/images/data/images/...'
        # Solution: if img_path is absolute -> use it; otherwise if the string
        # already starts with images_dir, treat it as a path; else join images_dir / img_path
        img_path_str = str(p)
        images_dir_str = str(self.images_dir)

        if not p.is_absolute():
            # normalize leading './' or '.\\' cases
            if img_path_str.startswith('./') or img_path_str.startswith('.\\'):
                img_path_str = img_path_str[2:]
            if img_path_str.startswith(images_dir_str + os.sep) or img_path_str == images_dir_str:
                p = Path(img_path_str)
            else:
                p = self.images_dir / p

        # optional: resolve symlinks / relative parts (don't force absolute to keep portability)
        p = p
        if not p.exists():
            # helpful debug message
            raise FileNotFoundError(f"Image not found: {p} (original image_path='{img_path}')")
        img = Image.open(p).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return img, label


def get_dataloaders(csv_path: str, images_dir: str, batch_size: int = 32,
                    val_ratio: float = 0.15, test_ratio: float = 0.15,
                    image_size: int = 224, seed: int = 42,
                    use_weighted_sampler: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].reset_index(drop=True)

    train_ds = ChartDataset(df_train, images_dir=images_dir, image_size=image_size, train=True)
    val_ds = ChartDataset(df_val, images_dir=images_dir, image_size=image_size, train=False)
    test_ds = ChartDataset(df_test, images_dir=images_dir, image_size=image_size, train=False)

    # WeightedRandomSampler 추가
    if use_weighted_sampler:
        labels = df_train['label'].values
        class_counts = np.bincount(labels.astype(int))
        weights = 1. / class_counts[labels.astype(int)]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_dl, val_dl, test_dl


def build_model(pretrained: bool = True, dropout1: float = 0.5, dropout2: float = 0.3) -> nn.Module:
    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # remove fc
    n_features = backbone.fc.in_features
    backbone.fc = nn.Identity()

    head = nn.Sequential(
        nn.Linear(n_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout1),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

    model = nn.Sequential(
        backbone,
        head
    )
    return model


def compute_loss_and_outputs(model, batch, device, criterion):
    imgs, labels = batch
    imgs = imgs.to(device)
    labels = labels.to(device)
    outputs = model(imgs).view(-1)
    loss = criterion(outputs, labels)
    return loss, outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()


def evaluate_model(model, dataloader, device) -> dict:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            out = model(imgs).view(-1).cpu().numpy()
            preds.extend(out.tolist())
            trues.extend(labels.numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)
    # metrics
    try:
        auc = roc_auc_score(trues, preds)
    except Exception:
        auc = float('nan')
    preds_bin = (preds >= 0.5).astype(int)
    acc = accuracy_score(trues, preds_bin)
    prec = precision_score(trues, preds_bin, zero_division=0)
    rec = recall_score(trues, preds_bin, zero_division=0)
    cm = confusion_matrix(trues, preds_bin)

    return {
        'auc': auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': cm,
        'n_pos': int(trues.sum()),
        'n_total': len(trues)
    }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    train_dl, val_dl, test_dl = get_dataloaders(
        args.data_csv,
        args.images_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        image_size=args.image_size,
        use_weighted_sampler=True  # 이 부분 추가!
    )

    model = build_model(pretrained=args.pretrained)
    model = model.to(device)

    # class imbalance analysis
    df = pd.read_csv(args.data_csv)
    pos = df['label'].sum()
    neg = len(df) - pos
    if pos == 0:
        pos = 1.0

    # Calculate alpha for Focal Loss (alpha balances positive/negative classes)
    # alpha should be inverse of class frequency for the positive class
    if args.focal_alpha is not None:
        alpha = args.focal_alpha
    else:
        alpha = neg / (pos + neg)

    print(f"pos:{pos}, neg:{neg}, focal_loss_alpha:{alpha:.3f}, focal_loss_gamma:{args.focal_gamma}")

    # Focal Loss with auto-calculated alpha and configurable gamma
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).view(-1)

            # Focal Loss
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)

        val_metrics = evaluate_model(model, val_dl, device)
        val_auc = val_metrics['auc'] if not np.isnan(val_metrics['auc']) else 0.0

        print(
            f"Epoch {epoch} | train_loss: {avg_loss:.4f} | val_auc: {val_auc:.4f} | val_acc: {val_metrics['accuracy']:.4f}")

        # save best
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'args': vars(args)},
                       os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print("Saved best model")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    # final test evaluation
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    test_metrics = evaluate_model(model, test_dl, device)
    print("Test metrics:", test_metrics)


def eval_mode(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_dl = get_dataloaders(
        args.data_csv,
        args.images_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        image_size=args.image_size,
        use_weighted_sampler=False  # eval에서는 불필요
    )
    model = build_model(pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    metrics = evaluate_model(model, test_dl, device)
    print(metrics)


def predict_single(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    img = Image.open(args.image).convert('RGB')
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(x).view(-1).item()
    print(f"probability: {prob:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True)
    p.add_argument('--data_csv', type=str, default='data/processed/labels.csv')
    p.add_argument('--images_dir', type=str, default='data/images')
    p.add_argument('--checkpoint_dir', type=str, default='models')
    p.add_argument('--checkpoint', type=str, default='models/best_model.pth')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--image', type=str, default=None)
    p.add_argument('--focal_alpha', type=float, default=None, help='Focal Loss alpha (auto-calculated if not provided)')
    p.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma (default: 2.0)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval_mode(args)
    elif args.mode == 'predict':
        if not args.image:
            raise ValueError('predict 모드에서는 --image 를 지정하세요')
        predict_single(args)
    else:
        raise ValueError('unknown mode')