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

def remove_background(image_path, new_path):
    """
    Removes the background of an image and save it with a 
    transparent background to a new folder.

    Args:
        image_path (str): The path to the image file.
        new_path (str): The path to the directory where the processed image
                        will be saved.

    Returns:
        None
    """

    input = Image.open(image_path)

    path_directories = [directory for directory in image_path.split('/')]
    image_name = path_directories[-1]
    print(image_name)

    output = remove(input)
    output.save(new_path + '/' + image_name, format='PNG')

def features_extraction(model, im_path):
    """
    Extracts features for images located within a specified directory path.

    Args:
        model (object): The pre-trained deep learning model used for feature extraction.
        im_path (str): The path to the directory containing the images.

    Returns:
        features_extracted (dict): A dictionary where keys are the image paths and values are the corresponding
                                   extracted feature tensors.
    """

    # Get all image file paths
    img_names = [os.path.join(folder, img) for folder, _, filenames in os.walk(im_path) for img in filenames]

    # Create dict with paths as keys and tensors of the images as values
    images = {path: Image.open(path).convert('RGB') for path in img_names}

    # Extract features and reshape
    features_extracted = {path: model.forward(images[path]) for path in images.keys()}

    return features_extracted

def similarity_extraction(input_image_path, features_extracted, method="cosine"):
    """
    Calculates the similarity between a given image and other images based on pre-extracted features.

    Args:
        input_image_path (str): The path to the image for which you want to find similar images.
        features_extracted (dict): A dictionary that maps image paths (str) to their corresponding 
                                    extracted feature tensors (object). This dictionary is returned 
                                    by the `features_extraction` function.
        method (str, optional): The method used to calculate similarity. 
                                    Defaults to "cosine". Supported methods include:
                                        - "cosine": Cosine similarity.
                                        - "euclidean": Euclidean distance (negative similarity).
                                        - "manhattan": Manhattan distance (negative similarity).

    Returns:
        sorted_simlarities (dict): A dictionary where keys are the image paths (str) from the `features_extracted` dictionary, 
                                     and values are the corresponding similarity scores (float) between those images and the 
                                     input image. The dictionary is sorted in descending order of similarity, with the most similar 
                                     images having the highest values.
    """

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

def ranking_similarities(dict_similarities, metrics, models, top_n=10):
    """
    Generates rankings of similar images based on pre-calculated similarities from different models and metrics.

    Args:
        dict_similarities (dict): A dictionary where keys are combinations of model names (str) and 
                                    metrics (str) concatenated with an underscore ('_'). Values are 
                                    dictionaries returned by the `similarity_extraction` function. 
                                    These inner dictionaries map image paths (str) to their 
                                    corresponding similarity scores (float).
        metrics (list): A list of strings representing the metrics used for similarity calculation 
                          ("cosine", "euclidean", "manhattan").
        models (list): A list of strings representing the different models used for feature extraction.
        top_n (int, optional): The maximum number of top similar images to include in each ranking.
                                Defaults to 10.

    Returns:
        ranks (list): A list of rankings, where each element is a sublist representing a ranking for a 
                        specific combination of model and metric. Each sublist contains the top-n most similar 
                        image paths (str) in descending order based on their similarity scores.
    """

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

def reciprocal_rank_fusion(ranks, k=0):
    """
    Performs reciprocal rank fusion (RRF) to combine multiple image rankings and 
    generate a single, fused ranking.

    Args:
        ranks (list): A list of rankings, where each element is a sublist representing 
                        a ranking for a specific model or metric. Each sublist contains 
                        image paths (str) in their order of similarity.
        k (int, optional): A constant used in the RRF scoring formula. Defaults to 0.

    Returns:
        sorted_scores (dict): A dictionary where keys are image paths (str) and values are their 
                                corresponding fused ranking scores (float). The dictionary is sorted in 
                                descending order of scores, meaning images with higher scores are ranked 
                                higher in the final, fused ranking.
    """
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
    

def borda_count(ranks):
    """
    Performs Borda Count to aggregate multiple image rankings and 
    generate a single, fused ranking.

    Args:
        ranks (list): A list of rankings, where each element is a sublist representing 
                        a ranking for a specific model or metric. Each sublist contains 
                        image paths (str) in their order of similarity.

    Returns:
        sorted_scores (dict): A dictionary where keys are image paths (str) and values are their 
                                corresponding Borda Count scores (int). The dictionary is sorted in 
                                descending order of scores, meaning images with higher scores are ranked 
                                higher in the final, fused ranking.
    """
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


def relative_score_fusion(sorted_sim_per_model, metrics, models):
    """
    Performs relative score fusion to combine multiple pre-sorted image similarity dictionaries 
    and generate a single, fused ranking.

    Args:
        sorted_sim_per_model (dict): A dictionary where keys are combinations of model names (str) 
                                        and metrics (str) concatenated with an underscore ('_'). 
                                        Values are dictionaries returned by the `similarity_extraction` 
                                        function after sorting them in descending order of similarity 
                                        (highest similarity scores first).
        metrics (list): A list of strings representing the metrics used for similarity calculation 
                        (e.g., "cosine", "euclidean", "manhattan").
        models (list): A list of strings representing the different models used for feature extraction.

    Returns:
        sorted_scores (dict): A dictionary where keys are image paths (str) and values are their 
                                corresponding fused ranking scores (float) calculated using relative score fusion. 
                                The dictionary is sorted in descending order of scores, meaning images with higher 
                                scores are ranked higher in the final, fused ranking.
    """
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

def plot_recommendations(sorted_similarities, input_image_path, model_name="", top_n=4):
    """
    Visualizes the top-n most similar images based on the provided ranked similarities.

    Args:
        sorted_similarities (dict): A dictionary where keys are image paths (str) and values are 
                                    their corresponding similarity scores (float). The dictionary 
                                    is assumed to be sorted in descending order of similarity scores.
        input_image_path (str): The path to the input image for which recommendations are shown.
        model_name (str, optional): The name of the model used for similarity calculation 
                                    (used for labeling the input image). Defaults to "".
        top_n (int, optional): The maximum number of top recommendations to visualize. Defaults to 4.

    Returns:
        None
    """
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
  
