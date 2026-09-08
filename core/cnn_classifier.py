from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .types import ClassificationResult


class CNNClassifier:
    def __init__(self, model_path: str | Path, class_names_path: str | Path, device: str = 'cpu'):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.device = device
        self._model = None
        self._transform = None
        self._class_names: list[str] = []

    def available(self) -> bool:
        return self.model_path.is_file() and self.class_names_path.is_file()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available():
            raise FileNotFoundError(
                f'CNN classifier files missing: {self.model_path} / {self.class_names_path}'
            )
        import torch
        from torchvision import models, transforms

        names = json.loads(self.class_names_path.read_text(encoding='utf-8'))
        if isinstance(names, dict):
            names = [name for name, _ in sorted(names.items(), key=lambda kv: int(kv[1]))]
        if not isinstance(names, list) or not names:
            raise ValueError('class_names.json must contain a non-empty list or name->index mapping')
        self._class_names = [str(x) for x in names]
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, len(self._class_names))
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self._model = model
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image: np.ndarray) -> ClassificationResult:
        start = time.perf_counter()
        self._load()
        import torch

        if image.ndim == 2:
            pil = Image.fromarray(image).convert('RGB')
        else:
            # Incoming arrays elsewhere in the project are OpenCV/BGR.
            pil = Image.fromarray(image[:, :, ::-1]).convert('RGB')
        tensor = self._transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, idx = torch.max(probs, dim=0)
        scores = {name: float(p.item()) for name, p in zip(self._class_names, probs)}
        return ClassificationResult(
            self._class_names[int(idx.item())],
            float(confidence.item()),
            'cnn',
            (time.perf_counter() - start) * 1000,
            scores=scores,
        )
