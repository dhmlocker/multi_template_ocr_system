from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.dataset import validate_imagefolder_layout


def resolve_backbone_source(
    pretrained_backbone: str | Path,
    *,
    no_pretrained: bool,
    allow_download: bool,
) -> str:
    """Choose pretrained source without silently touching the network."""
    if no_pretrained:
        return 'none'
    path = Path(pretrained_backbone)
    if path.is_file():
        return 'local'
    if allow_download:
        return 'torchvision'
    raise FileNotFoundError(
        f'local MobileNetV2 ImageNet weight not found: {path}. '
        'Put your downloaded mobilenet_v2_imagenet.pth there, use --no-pretrained, '
        'or explicitly pass --allow-torchvision-download.'
    )


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise ValueError('pretrained backbone file does not contain a PyTorch state_dict mapping')
    cleaned = {}
    for key, value in checkpoint.items():
        name = str(key)
        for prefix in ('module.', 'model.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def build_model(
    num_classes: int,
    *,
    pretrained_backbone: str | Path = 'models/mobilenet_v2_imagenet.pth',
    no_pretrained: bool = False,
    allow_download: bool = False,
):
    import torch
    from torchvision import models

    source = resolve_backbone_source(
        pretrained_backbone,
        no_pretrained=no_pretrained,
        allow_download=allow_download,
    )
    if source == 'torchvision':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        model = models.mobilenet_v2(weights=None)
        if source == 'local':
            try:
                checkpoint = torch.load(Path(pretrained_backbone), map_location='cpu', weights_only=True)
            except TypeError:  # older PyTorch
                checkpoint = torch.load(Path(pretrained_backbone), map_location='cpu')
            incoming = _extract_state_dict(checkpoint)
            current = model.state_dict()
            compatible = {
                k: v for k, v in incoming.items()
                if k in current and getattr(v, 'shape', None) == current[k].shape
            }
            if not compatible:
                raise ValueError(
                    'The local MobileNetV2 pretrained file has no compatible torchvision MobileNetV2 parameters.'
                )
            model.load_state_dict(compatible, strict=False)
            print(f'Loaded {len(compatible)} compatible parameters from local backbone: {pretrained_backbone}')

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description='Train MobileNetV2 template classifier using user-provided images only.')
    parser.add_argument('--train-dir', default='data/train')
    parser.add_argument('--val-dir', default='data/val')
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pretrained-backbone', default='models/mobilenet_v2_imagenet.pth', help='Local torchvision MobileNetV2 ImageNet state_dict.')
    parser.add_argument('--no-pretrained', action='store_true', help='Train the classification head without ImageNet pretrained backbone weights.')
    parser.add_argument('--allow-torchvision-download', action='store_true', help='Explicitly permit torchvision to download ImageNet weights when local file is missing.')
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    train_dir, val_dir = Path(args.train_dir), Path(args.val_dir)
    train_classes = validate_imagefolder_layout(train_dir)
    val_classes = validate_imagefolder_layout(val_dir)
    if train_classes != val_classes:
        raise ValueError(f'train/val class folders differ: {train_classes} vs {val_classes}')

    tfm_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=tfm_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(
        len(train_classes),
        pretrained_backbone=args.pretrained_backbone,
        no_pretrained=args.no_pretrained,
        allow_download=args.allow_torchvision_download,
    ).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    best_acc = -1.0
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics = []
    for epoch in range(1, args.epochs + 1):
        model.train(); correct = total = 0; train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(); logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            train_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item()); total += y.size(0)
        model.eval(); val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                val_correct += int((logits.argmax(1) == y).sum().item()); val_total += y.size(0)
        row = {'epoch': epoch, 'train_loss': train_loss/max(total,1), 'train_acc': correct/max(total,1), 'val_acc': val_correct/max(val_total,1)}
        metrics.append(row); print(row)
        if row['val_acc'] > best_acc:
            best_acc = row['val_acc']; torch.save(model.state_dict(), out/'classifier.pth')
    (out/'class_names.json').write_text(json.dumps(train_classes, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'training_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Best validation accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()
