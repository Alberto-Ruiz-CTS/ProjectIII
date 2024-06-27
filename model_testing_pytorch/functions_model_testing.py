
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from scipy.spatial.distance import cosine
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


def features_extraction(model, im_path, size=224):
    #define transformations to preprocess images
    preprocess = transforms.Compose([     
            transforms.Resize(size),             # resize shortest side to specified pixels
            transforms.CenterCrop(size),         # crop longest side to specified pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    
    img_names = []

    #Create list with all the paths to the images
    for folder, subfolders, filenames in os.walk(im_path):
        for img in filenames:
            img_names.append(folder+'/'+img)

    try:
        img_names.remove("/teamspace/studios/this_studio/ProjectIII/women_fashion.DS_Store") #remove DS_Store if it is in the folder
    except:
        pass

    #Create dict with paths as keys and tensors of the images as values
    images = {} 

    for path in img_names: 

        try:
            image = preprocess(Image.open(path))
            images[path] = image
        
        except:
            pass #prevent any error with particular images

    #inception_v3_model = models.model(pretrained=True) #Initialize model

    for param in model.parameters(): #Freeze all parameters of the model
        param.requires_grad = False

    #Create dict with paths as keys and features (numpy array) as values
    all_features = {}

    model.eval()
    with torch.no_grad():
        for path in images.keys():
            feature = model(images[path].view(1,3,size,size)).numpy()
            all_features[path] = np.reshape(feature, feature.shape[1])

    return all_features




def similarity_extraction(input_image_path, all_features, method="cosine"):

    if method not in ("cosine", "euclidean"):

        raise ValueError("Method not valid")

    input_features = all_features[input_image_path]
    
    similarities = {} #Dict with path as keys and similarity as values

    if method == "cosine":
        for path in all_features:
            similarities[path] = 1 - cosine(input_features, all_features[path])
    else:
        raise Exception("Method not implemented yet")

    #sort this similarities dict by their similarities from largest to smallest
    keys = list(similarities.keys())
    values = list(similarities.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_similarities = {keys[i]: values[i] for i in sorted_value_index}
    

    return sorted_similarities
    

def plot_recommendations(sorted_similarities, model_name="",top_n=4):

    #Create list with the paths for the top_n most similar images
    sorted_paths = []

    for i, path in enumerate(sorted_similarities):
        if i <= top_n:
            sorted_paths.append(path)
        else: 
            break
    
    input_image_path = sorted_paths[0]
    # display the input image
    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title(f"Input Image {model_name}")
    plt.axis('off')

    # display similar images
    for i, path in enumerate(sorted_paths[1:], start=1):
        image_path = path
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"Recommendation {i}")
        plt.axis('off')


    
