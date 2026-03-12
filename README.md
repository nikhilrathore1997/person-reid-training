# Custom Person Re-ID Model
ResNet-50 + ArcFace Head + BatchHard Triplet Loss — trained for production edge deployment

## Architecture
- Backbone: ResNet-50 (pretrained ImageNet)
- Embedding head: Linear → BatchNorm → Dropout (512-dim)
- Loss: BatchHard Triplet Loss + CrossEntropy
- Sampler: RandomIdentitySampler (P=32 identities, K=4 images)

## Optimization Pipeline
PyTorch → TensorRT (GPU) → OpenVINO INT8 (CPU/Edge) → Raspberry Pi @ 4 FPS

## Dataset
Market-1501 style structure: /class_name/image.jpg

## Train
```bash
pip install torch torchvision
python train.py
```
