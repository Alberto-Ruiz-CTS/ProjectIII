import torch
from torchvision import models, transforms
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from PIL import Image as PIL_Image

class FeaturesExtractorModel(nn.Module):
    
    def __init__(self, model_name="vgg16"):
        super(FeaturesExtractorModel, self).__init__()

        # Load pre-trained model based on name
        self.available_models = {'vgg16': models.vgg16(models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)),
                                  'resnet50': models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
                                  'googlenet': models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1),
                                  'efficient_net_b0': models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1),
                                  'vit_b_16': vit_b_16(weights=ViT_B_16_Weights.DEFAULT)}

        # Set transform to be used
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Get the models that will be used to encode images into features
        self.model_name = model_name
        self.model = self.available_models[self.model_name]
        self.get_model()

    def forward(self, x):

        x = self.transform(x)
        x = x.unsqueeze(0)

        # Extract features based on the model type
        if self.model_name == 'vit_b_16':
            features = self.extract_features_vit_b_16(x)
        else:
            features = self.model(x)

        features = features.detach().numpy().reshape(-1)

        return features

    def get_model(self):
        # All of the models except vgg16 share layer extraction procedure
        if self.model_name == 'vgg16':
            self.model.classifier = self.model.classifier[:-1]
        elif self.model_name != 'vit_b_16':
            self.model = nn.Sequential(*list(self.model.children())[:-1])

    def extract_features_vit_b_16(self, img):

        features = self.model._process_input(img)

        # Expand the CLS token to the full batch
        batch_class_token = self.model.class_token.expand(img.shape[0], -1, -1)
        features = torch.cat([batch_class_token, features], dim=1)
        features = self.model.encoder(features)

        # We're only interested in the representation of the CLS token that we appended at position 0
        features = features[:, 0]
        features = features.squeeze()

        return features