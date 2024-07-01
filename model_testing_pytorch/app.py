import streamlit as st
from PIL import Image
import paths as p
import functions_model_testing as f
import os
from rembg import remove
from torchvision import models



def upload_and_display():
  size = 224

  uploaded_image = st.file_uploader("Upload Image")
  if uploaded_image is not None:
    filename = uploaded_image.name
    image = Image.open(uploaded_image)

    image = remove(image)
    input_path = p.new_path + '/' + filename
    inside_data = False
    #Walk through the folder of the images and save it in the new folder without background
    for folder, subfolders, filenames in os.walk(p.new_path):
      for img in filenames:
        if os.path.isfile(input_path) == False:
          image.save(input_path, format='PNG')
        else:
          inside_data = True
    
    st.image(image, width=400)  # Adjust width as needed

    #dict with models that are going to be tested and the image size they require as a tuple (model, size) (Different models may require different image sizes)
    test_models = {
        #"resnet": (models.resnet18(pretrained=True), 224),
        #"alexnet": (models.alexnet(pretrained=True), 224),
        "vgg16": (models.vgg16(pretrained=True), 224),
        "densenet": (models.densenet161(pretrained=True), 224),
        "inception": (models.inception_v3(pretrained=True), 299),
        #"googlenet": (models.googlenet(pretrained=True), 224),
    }
        
    #Create a dict with the features for each model and metric combination
    sorted_sim_per_model = {}
    for model_name in test_models:
        for metrics in ['cosine', 'euclidean', 'manhattan']:
    
            model, size = test_models[model_name]
            features = f.features_extraction(model, p.new_path, size=size)
            sorted_similarities = f.similarity_extraction(input_path, features, method=metrics)
            sorted_sim_per_model[model_name + '_' + metrics] = sorted_similarities
      
    #Rank the best recommendations for each model
    ranks = f.ranking_similarities(sorted_sim_per_model, top_n=10)

    #Create a rank fusion of all individuals rankings
    final_dict = f.reciprocal_rank_fusion(ranks)
        
    # Placeholder recommendation logic (replace with your actual logic)
    recommended_images = []

    for i, key in enumerate(final_dict.keys()):
      if i < 3 and inside_data == False:
        recommended_images.append(key)
      elif inside_data == True and i != 0 and i < 4:
        recommended_images.append(key)

    #recommended_images = []

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
