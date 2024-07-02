from scipy.spatial.distance import cosine, euclidean, cityblock
import numpy as np

################# updated ##################
# Reciprocal Rank Fusion
def reciprocal_rank_fusion(ranks, k=0):
    """ Fusion of the different ranks the user provides and returns a sorted dictionary with the best image paths and their ranking scores """

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


def ranking_similarities(dict_similarities, metrics, models, top_n=10):
    """ Create a list of the rankings for each model """

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

###################################################################################
def reciprocal_rank_fusion(ranks, k=0):
    n   = len(ranks[0])
    scores = np.zeros(n)
    for rank in ranks:
        for i, idx in enumerate(rank):
            scores[idx] += 1 / (k + i)
    return np.argsort(scores)[::-1]

def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    # Preprocess the input image and extract features
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    # Calculate similarity scores using multiple metrics
    cosine_scores = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    euclidean_scores = [1 - euclidean(input_features, other_feature) for other_feature in all_features]
    manhattan_scores = [1 - cityblock(input_features, other_feature) for other_feature in all_features]

    # Rank the images based on each metric
    cosine_ranks = np.argsort(cosine_scores)[::-1]
    euclidean_ranks = np.argsort(euclidean_scores)[::-1]
    manhattan_ranks = np.argsort(manhattan_scores)[::-1]

    # Filter out the input image index from the ranks
    input_idx = all_image_names.index(input_image_path)
    cosine_ranks = [idx for idx in cosine_ranks if idx != input_idx]
    euclidean_ranks = [idx for idx in euclidean_ranks if idx != input_idx]
    manhattan_ranks = [idx for idx in manhattan_ranks if idx != input_idx]

    # Apply Borda count for rank fusion
    all_ranks = [cosine_ranks, euclidean_ranks, manhattan_ranks]
    final_ranks = borda_count(all_ranks)

    # Get the top N recommendations
    top_n_recommendations = final_ranks[:top_n]

    # Return the names of the top N recommended images
    return [all_image_names[idx] for idx in top_n_recommendations]

