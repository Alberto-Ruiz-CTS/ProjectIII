import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from PIL import Image
from rembg import remove
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from scipy.spatial.distance import cosine, euclidean, cityblock
import torch.nn as nn

# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

#Remove the background of an image
def remove_background(image_path, new_path):
    """ Remove the background of the images and save it in a new folder """

    input = Image.open(image_path)

    path_directories = [directory for directory in image_path.split('/')]
    image_name = path_directories[-1]
    print(image_name)

    output = remove(input)
    output.save(new_path + '/' + image_name, format='PNG')

'''
#Extract the features of an image
def features_extraction(model, im_path, size=224):
    """ Search the paths for the images, preprocess the data and return the features for an image """

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
            image = preprocess(Image.open(path).convert('RGB'))
            images[path] = image
        
        except:
            pass #prevent any error with particular images

    #inception_v3_model = models.model(pretrained=True) #Initialize model

    for param in model.parameters(): #Freeze all parameters of the model
        param.requires_grad = False

    #Create dict with paths as keys and features (numpy array) as values
    features_extracted = {}

    model.eval()
    with torch.no_grad():
        for path in images.keys():
            feature = model(images[path].view(1,3,size,size)).numpy()
            features_extracted[path] = np.reshape(feature, feature.shape[1])

    return features_extracted

#Drop fully connected layer from the model to retain only th encoder
def get_encoder(model_name):

    match model_name: #each model requires a different action depending
        case 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier = model.classifier[:-1]
            return model
        case 'resnet50':
            model = models.resnet50(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            return model
        case 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            return model
        case 'googlenet':
            model = models.googlenet(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            return model
'''

#Extract the features of an image
def features_extraction(model, im_path):
    """ Search the paths for the iamges, preprocess the data and return the features for an image """

    # Get all image file paths
    img_names = [os.path.join(folder, img) for folder, _, filenames in os.walk(im_path) for img in filenames]
    # img_names.remove("/teamspace/studios/this_studio/ProjectIII/women_fashion.DS_Store")

    #Create dict with paths as keys and tensors of the images as values
    images = {path: Image.open(path).convert('RGB') for path in img_names}

    # Extract features and reshape
    features_extracted = {path: model.forward(images[path]) for path in images.keys()}

    return features_extracted

#Give the similarity between to images
def similarity_extraction(input_image_path, features_extracted, method="cosine"):
    """ Calculate the similarity of two images with the method provides in input and returns a sorted dictionary """

    input_features = features_extracted[input_image_path]
    similarities = {} #Dict with path as keys and similarity as values

    if method == "cosine":
        for path in features_extracted:
            similarities[path] = 1 - cosine(input_features, features_extracted[path])
    elif method == "euclidean":
        for path in features_extracted:
            similarities[path] = - euclidean(input_features, features_extracted[path])
    elif method == "manhattan":
        for path in features_extracted:
            similarities[path] = - cityblock(input_features, features_extracted[path])

    # Sort this similarities dict by their similarities from largest to smallest
    keys = list(similarities.keys())
    values = list(similarities.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_similarities = {keys[i]: values[i] for i in sorted_value_index}
    
    return sorted_similarities

#Give the similarity between to images
def similarity_extraction(input_image_path, features_extracted, method="cosine"):
    """ Calculate the similarity of two images with the method provides in input and returns a sorted dictionary """

    if method not in ("cosine", "euclidean", "manhattan"):

        raise ValueError("Method not valid")

    input_features = features_extracted[input_image_path]
    
    similarities = {} #Dict with path as keys and similarity as values

    if method == "cosine":
        for path in features_extracted:
            similarities[path] = 1 - cosine(input_features, features_extracted[path])
    elif method == "euclidean":
        for path in features_extracted:
            similarities[path] = - euclidean(input_features, features_extracted[path])
    elif method == "manhattan":
        for path in features_extracted:
            similarities[path] = - cityblock(input_features, features_extracted[path])
    else:
        raise Exception("Method not implemented yet")

    #sort this similarities dict by their similarities from largest to smallest
    keys = list(similarities.keys())
    values = list(similarities.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_similarities = {keys[i]: values[i] for i in sorted_value_index}
    
    return sorted_similarities

#Make a rank of each model
def ranking_similarities(dict_similarities, metrics, models, top_n=10):
    """ Create a list of the rankings for each model """

    dict_sim = {}
    
    for model in models:
        for metric in metrics:
            dict_sim[model + '_' + metric] = dict_similarities[model + '_' + metric]
    
    ranks = []
 
    for models in dict_sim:
        rank = []
        for i, path in enumerate(dict_sim[models]):
            if (i != 0) and (i <= top_n):
                rank.append(path)
 
        ranks.append(rank)
   
    return ranks

# Recirpocal Rank Fusion
def reciprocal_rank_fusion(ranks, k=0):
    """ Fusion of the differents ranks the user provides and returns a sorted dictionary with the best image paths and their ranking scores """

    n = len(ranks[0])
    scores = {}
    for rank in ranks:
        for i, path in enumerate(rank, start=1):
            if path in scores.keys():
                scores[path] += 1 / (k + i)
            else:
                scores[path] = 1 / (k + i)
    
    #sort this scores dict by their score values from largest to smallest
    keys = list(scores.keys())
    values = list(scores.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_scores = {keys[i]: values[i] for i in sorted_value_index}

    return sorted_scores
    

# Borda Algorithm
def borda_count(ranks):
    """ Broda algorithm of the different ranks the user provides and returns a sorted dictionary with the best image paths and their ranking scores """

    n = len(ranks[0])
    scores = {}
    for rank in ranks:
        for i, path in enumerate(rank):
            if path in scores.keys():
                scores[path] += (n - i)
            else:
                scores[path] = (n - i)

    #sort this scores dict by their score values from largest to smallest
    keys = list(scores.keys())
    values = list(scores.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_scores = {keys[i]: values[i] for i in sorted_value_index}

    return sorted_scores


# Relative Score Fusion
def relative_score_fusion(sorted_sim_per_model, metrics, models):
    """ Fusion of the different ranks the user provides and returns a sorted dictionary with the best image paths and their ranking scores """

    # normalize similiarities
    norm_similarity = {}
    for model_name in models:
        for metric in metrics:
            total_sim = sum(sorted_sim_per_model[model_name + '_' + metric].values())
            norm_similarity[model_name + '_' + metric] = {}
            for path in sorted_sim_per_model[model_name + '_' + metric]:
                norm_similarity[model_name + '_' + metric][path] = sorted_sim_per_model[model_name + '_' + metric][path] / total_sim

    scores = {}
    for method in norm_similarity:
        for path in norm_similarity[method]:
            if path in scores.keys():
                scores[path] += norm_similarity[method][path]
            else:
                scores[path] = norm_similarity[method][path]

    #sort this scores dict by their score values from largest to smallest
    keys = list(scores.keys())
    values = list(scores.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_scores = {keys[i]: values[i] for i in sorted_value_index}

    return sorted_scores

#Plot the best recommendations for the input image
def plot_recommendations(sorted_similarities, input_image_path, model_name="", top_n=4):
    """ Plot the best k recommendations for the input image """

    #Create list with the paths for the top_n most similar images
    sorted_paths = []

    for i, path in enumerate(sorted_similarities):
        if i <= top_n:
            sorted_paths.append(path)
        else: 
            break
    
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
  
