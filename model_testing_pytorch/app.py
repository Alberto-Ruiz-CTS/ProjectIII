import streamlit as st
from PIL import Image
import paths as p
import functions_model_testing as f
from model import FeaturesExtractorModel
import os
from rembg import remove
from torchvision import models
import pickle

st.set_page_config(layout="wide")

def upload_and_display():
  size = 224
  inside_data = False
  sorted_sim_per_model = {}
  final_dict = {}
  recommended_images = []
  default_models = ['vgg16']
  default_metrics = ['cosine']
  opt_models = ['vgg16','resnet50','efficient_net_b0']
  opt_metrics = ['cosine', 'euclidean', 'manhattan']
  opt_fusion = ['reciprocal', 'borda', 'relative score']

  dress_title, rec1_title, rec2_title, rec3_title = st.columns(4, vertical_alignment="top") 
  dress, rec1, rec2, rec3 = st.columns(4, vertical_alignment="top") 

  with st.sidebar:
    fusion_method = st.selectbox("Select a fusion method:", opt_fusion, key="select_rank")

    metrics = st.multiselect("Select one or more models:", opt_metrics, default=default_metrics, key="mselect_metrics")

    test_models = st.multiselect("Select one or more models:", opt_models, default=default_models, key="mselect_models")

    background = st.checkbox("Remove background")


  uploaded_image = st.file_uploader("Upload Image")
  if uploaded_image is not None:
    filename = uploaded_image.name
    image = Image.open(uploaded_image)

    #Remove background checbox functionality
    if background:
      image = remove(image)
      path = p.new_path
      input_path = p.new_path + '/' + filename
      ext_path = p.ext_path + '_rb'
      ext_img_path = ext_path + '/' + filename
      pkl = 'features_data_rb.pkl'   
    else:
      path = p.data_path
      input_path = p.data_path + '/' + filename
      ext_path = p.ext_path
      ext_img_path = p.ext_path + '/' + filename
      pkl = 'features_data.pkl'
      
    #Walk through the folder of the images and save it in the new folder without background
    for folder, subfolders, filenames in os.walk(path):
      for img in filenames:
        if os.path.isfile(input_path) == False and os.path.isfile(ext_img_path) == False:
          image.save(ext_img_path, format='PNG')
        elif os.path.isfile(input_path):
          inside_data = True
    
    image = image.resize((size, size))
    dress_title.subheader("Your Dress")  # Add title or description 
    dress.image(image, width=size)  # Adjust width as needed

    # Read dictionary pkl file
    with open(p.dict_path + '/' + pkl, 'rb') as fp:
      feats = pickle.load(fp)

    for model_name in test_models:
      if input_path not in feats[model_name].keys() and ext_img_path not in feats[model_name].keys(): 
        featuresExtractorModel = FeaturesExtractorModel(model_name)
        input_feature = f.features_extraction(featuresExtractorModel, ext_path)
        feats[model_name].update(input_feature)

        # save dictionary to person_data.pkl file
        with open(p.dict_path + '/' + pkl, 'wb') as fp:
          pickle.dump(feats, fp)

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
      if i < 3 and inside_data == False or inside_data == True and i != 0 and i < 4:
        recommended_images.append(key)

    for i, recommended_image in enumerate(recommended_images):
      rec = [dress, rec1, rec2, rec3][i+1]
      rec_title = [dress_title, rec1_title, rec2_title, rec3_title][i+1]
      try:
        # Attempt to open the recommended image (handle potential errors)
        rec_image = Image.open(recommended_image)
        rec_image = rec_image.resize((size, size))
        rec_title.subheader(f"Recommendation {i+1}")  # Add title or description 
        rec.image(rec_image, width=size)  # Adjust width as needed

        #col.image(rec_image, width=200)
      except FileNotFoundError:
        rec.write(f"Recommended Image {i+1} Not Found")

st.title("Image Recommendation System")
upload_and_display()
