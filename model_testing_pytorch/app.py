import streamlit as st
from PIL import Image
import paths as p
import functions_model_testing as f
from model import FeaturesExtractorModel
import os
from rembg import remove
from torchvision import models
import pickle

def upload_and_display():
  size = 224
  final_dict = {}
  recommended_images = []
  default_models = ['vgg16']
  default_metrics = ['cosine']
  opt_models = ['vgg16','resnet50','efficient_net_b0']
  opt_metrics = ['cosine', 'euclidean', 'manhattan']
  opt_fusion = ['reciprocal', 'borda', 'relative score']

  # Create the columns
  col1, col2, col3 = st.columns(3)

  # Create the selectbox in the first column
  with col1:
    fusion_method = st.selectbox("Select an option:", opt_fusion, key="select_rank")

  # Create the multiselect box in the second column
  with col2:
    metrics = st.multiselect("Select one or more options:", opt_metrics, default=default_metrics, key="mselect_metrics")

  # Create the multiselect box in the second column
  with col3:
    test_models = st.multiselect("Select one or more options:", opt_models, default=default_models, key="mselect_models")


  uploaded_image = st.file_uploader("Upload Image")
  if uploaded_image is not None:
    filename = uploaded_image.name
    image = Image.open(uploaded_image)

    image = remove(image)
    input_path = p.new_path + '/' + filename
    ext_img_path = p.ext_path + '/' + filename
    inside_data = False
    #Walk through the folder of the images and save it in the new folder without background
    for folder, subfolders, filenames in os.walk(p.new_path):
      for img in filenames:
        if os.path.isfile(input_path) == False and os.path.isfile(ext_img_path) == False:
          image.save(ext_img_path, format='PNG')
        elif os.path.isfile(input_path):
          inside_data = True
    st.image(image, width=200)  # Adjust width as needed

    # Read dictionary pkl file
    with open(p.dict_path + '/' + 'features_data.pkl', 'rb') as fp:
        feats = pickle.load(fp)

    #Create a dict with the features for each model and metric combination
    sorted_sim_per_model = {}
    for model_name in test_models:
      if input_path in feats[model_name]: 
        pass
      else:
        featuresExtractorModel = FeaturesExtractorModel(model_name)
        input_feature = f.features_extraction(featuresExtractorModel, p.ext_path)
        feats[model_name].update(input_feature)

      for metric in metrics:
        if inside_data == True:
          sorted_similarities = f.similarity_extraction(input_path, feats[model_name], method=metric)
        else:
          sorted_similarities = f.similarity_extraction(ext_img_path, feats[model_name], method=metric)
        sorted_sim_per_model[model_name + '_' + metric] = sorted_similarities

    if fusion_method != 'relative score':
      #Rank the best recommendations for each model
      ranks = f.ranking_similarities(sorted_sim_per_model, metrics, test_models, top_n=10)

    if fusion_method == 'reciprocal':
      #Create a rank fusion of all individuals rankings
      final_dict = f.reciprocal_rank_fusion(ranks)
    elif fusion_method == 'borda':
      #Create a rank fusion of all individuals rankings
      final_dict = f.borda_count(ranks)
    elif fusion_method == 'relative score':
      # Apply Relative Score Fusion to combine different methods
      final_dict = f.relative_score_fusion(sorted_sim_per_model, metrics, test_models)

    for i, key in enumerate(final_dict.keys()):
      if i < 3 and inside_data == False:
        recommended_images.append(key)
      elif inside_data == True and i != 0 and i < 4:
        recommended_images.append(key)

    col1, col2, col3 = st.columns(3)
    for i, recommended_image in enumerate(recommended_images):
      col = [col1, col2, col3][i]
      try:
        # Attempt to open the recommended image (handle potential errors)
        rec_image = Image.open(recommended_image)
        col.image(rec_image, width=200)
      except FileNotFoundError:
        col.write(f"Recommended Image {i+1} Not Found")

st.title("Image Recommendation System")
upload_and_display()
