print("Hola")


from scipy.spatial.distance import cosine, euclidean, cityblock
import numpy as np

def borda_count(ranks):
    n = len(ranks[0])
    points = np.zeros(n)
    for rank in ranks:
        for i, idx in enumerate(rank):
            points[idx] += (n - i)
    return np.argsort(points)


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
