import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models


CLASS_NAMES = ['inaction', 'move', 'work']

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.cnn = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.cnn.classifier = nn.Identity()
        self.lstm = nn.LSTM(1280, 256, batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)

def load_frames(folder_path):
    folder = Path(folder_path)
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    frame_paths = []
    for ext in extensions:
        frame_paths.extend(folder.glob(ext))
    
    frame_paths = sorted(list(set(frame_paths)))
    
    if len(frame_paths) < 8:
        raise ValueError(f"Ошибка: В папке должно быть как минимум 8 кадров. Найдено: {len(frame_paths)}")
    

    frame_paths = frame_paths[:8]
    
    frames = [cv2.imread(str(p)) for p in frame_paths]
    return [f for f in frames if f is not None]

def preprocess_frames(frames, transform):
    processed = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb)
        processed.append(frame_tensor)
    return torch.stack(processed).unsqueeze(0)

def predict_single_with_confidence(model, frames, device, transform):

    video_tensor = preprocess_frames(frames, transform).to(device)
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), probabilities.detach().cpu().numpy()[0], confidence.item()

def advanced_tta(model, frames, device, transform):

    all_probs = []
    confidences = []


    _, probs, conf = predict_single_with_confidence(model, frames, device, transform)
    all_probs.append(probs)
    confidences.append(conf)


    flipped = [cv2.flip(f, 1) for f in frames]
    _, probs, conf = predict_single_with_confidence(model, flipped, device, transform)
    all_probs.append(probs)
    confidences.append(conf)


    flipped_v = [cv2.flip(f, 0) for f in frames]
    _, probs, conf = predict_single_with_confidence(model, flipped_v, device, transform)
    all_probs.append(probs)
    confidences.append(conf)


    rotated_90 = [cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames]
    _, probs, conf = predict_single_with_confidence(model, rotated_90, device, transform)
    all_probs.append(probs)
    confidences.append(conf)

    bright = [np.clip(f * 1.3, 0, 255).astype(np.uint8) for f in frames]
    _, probs, conf = predict_single_with_confidence(model, bright, device, transform)
    all_probs.append(probs)
    confidences.append(conf)

    dark = [np.clip(f * 0.7, 0, 255).astype(np.uint8) for f in frames]
    _, probs, conf = predict_single_with_confidence(model, dark, device, transform)
    all_probs.append(probs)
    confidences.append(conf)


    contrast_high = [np.clip(128 + 1.3 * (f - 128), 0, 255).astype(np.uint8) for f in frames]
    _, probs, conf = predict_single_with_confidence(model, contrast_high, device, transform)
    all_probs.append(probs)
    confidences.append(conf)

    blurred = [cv2.GaussianBlur(f, (5, 5), 0) for f in frames]
    _, probs, conf = predict_single_with_confidence(model, blurred, device, transform)
    all_probs.append(probs)
    confidences.append(conf)


    conf_weights = np.array(confidences) / np.sum(confidences)
    weighted_avg_probs = np.average(all_probs, weights=conf_weights, axis=0)

    return weighted_avg_probs

def main():
    parser = argparse.ArgumentParser(description="Инференс модели классификации действий.")
    parser.add_argument("folder", type=str, help="Путь к папке с 8 изображениями")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        frames = load_frames(args.folder)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    weights_dir = Path("/app/weights")
    models_pool = []
    
    for i in range(1, 6):
        weight_path = weights_dir / f"model_fold_{i}.pth"
        if weight_path.exists():
            model = VideoClassifier(num_classes=3).to(device)

            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            models_pool.append(model)
    
    if not models_pool:
        print("Ошибка: В папке /app/weights/ не найдено файлов весов фолдов (model_fold_1.pth ... model_fold_5.pth)")
        sys.exit(1)


    all_model_tta_probs = []
    for model in models_pool:
        model_tta_prob = advanced_tta(model, frames, device, transform)
        all_model_tta_probs.append(model_tta_prob)


    final_probs = np.average(all_model_tta_probs, weights=np.ones(len(models_pool))/len(models_pool), axis=0)
    final_class_idx = np.argmax(final_probs)
    

    print(CLASS_NAMES[final_class_idx])

if __name__ == "__main__":
    main()
