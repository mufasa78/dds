import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNetB4Detector(nn.Module):
    """
    Custom EfficientNet B4 model for deepfake detection
    """
    def __init__(self, num_classes=1):
        super(EfficientNetB4Detector, self).__init__()
        
        # Load EfficientNet B4 model without pre-trained weights for stability
        # Using standard torchvision model as a base
        self.backbone = models.efficientnet_b4(weights=None)
        
        # Replace the classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=1792, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Get features from backbone
        features = self.backbone.features(x)
        
        # Apply attention
        attention_mask = self.attention(features)
        features = features * attention_mask
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(features, (1, 1))
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class MesoNet(nn.Module):
    """
    Implementation of MesoNet for deepfake detection
    A simpler model that can be trained from scratch with less data
    """
    def __init__(self, num_classes=1):
        super(MesoNet, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third conv block
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth conv block
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4, 4)
        
        # Adaptive pooling to ensure consistent size regardless of input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Classifier
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Apply adaptive pooling to ensure correct size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
