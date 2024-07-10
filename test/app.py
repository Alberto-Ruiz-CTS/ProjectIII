import sys
sys.path.insert(1, '/teamspace/studios/this_studio/ProjectIII/src')
import config as p
import utils as f
from model import FeaturesExtractorModel

import os
from glob import glob
from rembg import remove
import streamlit as st
from PIL import Image
import pickle
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

st.set_page_config(layout="wide")

def sidebar():
  default_models = ['vgg16']
  default_metrics = ['cosine']
  default_fusion = ['reciprocal']
  opt_models = ['vgg16','resnet50','efficient_net_b0', 'vit_b_16', 'alexnet', 'googlenet']
  opt_metrics = ['cosine', 'euclidean', 'manhattan']
  opt_fusion = ['reciprocal', 'borda', 'relative score']
  opt = {}


  with st.sidebar:

    st.header("CNN options: ")

    opt["fusion_method"] = st.selectbox("Select a fusion method:", opt_fusion, key="select_rank")

    opt["metrics"] = st.multiselect("Select one or more metrics:", opt_metrics, default=default_metrics, key="mselect_metrics")

    opt["test_models"] = st.multiselect("Select one or more models:", opt_models, default=default_models, key="mselect_models")

    opt["background"] = st.checkbox("Remove background")

    if opt['metrics'] is None:
      st.write("Select a metric to calculate the similarity")
    if opt['test_models'] is None:
      st.write("Select a model to calculate the similarity")

  return opt

def upload(background):
  path = {}
  path["inside_data"] = False

  uploaded_image = st.file_uploader("Upload Image")
  if uploaded_image is not None:
    filename = uploaded_image.name
    image = Image.open(uploaded_image)

    #Remove background checbox functionality
    if background:
      #image = remove(image)
      path["path"] = p.new_path
      path["input_path"] = p.new_path + '/' + filename
      path["ext_path"] = p.ext_path + '_no_bg'
      path["ext_img_path"] = path["ext_path"] + '/' + filename
      path["pkl"] = 'features_data_rb.pkl'   
    else:
      path["path"] = p.data_path
      path["input_path"] = p.data_path + '/' + filename
      path["ext_path"] = p.ext_path
      path["ext_img_path"] = p.ext_path + '/' + filename
      path["pkl"] = 'features_data.pkl'
      
    #Walk through the folder of the images and save it in the new folder without background
    for folder, subfolders, filenames in os.walk(path["path"]):
      for img in filenames:
        if os.path.isfile(path["input_path"]) == False and os.path.isfile(path["ext_img_path"]) == False:
          if background:
            image = remove(image)
          image.save(path["ext_img_path"], format='PNG')
        elif os.path.isfile(path["input_path"]):
          path["inside_data"] = True

    return path
 
def similarities(opt, path, feats):

  input_feature = {}
  sorted_sim_per_model = {}

  for model_name in opt["test_models"]:
    if path['inside_data'] == False:  
    # if path["input_path"] not in feats[model_name].keys() and path["ext_img_path"] not in feats[model_name].keys(): 
      featuresExtractorModel = FeaturesExtractorModel(model_name)
      ext_feature = f.features_extraction(featuresExtractorModel, path["ext_path"])
      input_feature[path['ext_img_path']] = ext_feature[path['ext_img_path']]
      feats[model_name].update(input_feature)

    for metric in opt["metrics"]:
      if path["inside_data"] == True:
        sorted_similarities = f.similarity_extraction(path["input_path"], feats[model_name], method=metric)
      else:
        sorted_similarities = f.similarity_extraction(path["ext_img_path"], feats[model_name], method=metric)
      sorted_sim_per_model[model_name + '_' + metric] = sorted_similarities

  if path['inside_data'] == False:   
    for model_name in opt["test_models"]:
      feats[model_name].pop(path['ext_img_path'])

  return sorted_sim_per_model

def fusion_method(fusion_method, sorted_sim_per_model, metrics, test_models):
  try:
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

  except:
    final_dict = {}
    st.warning('Introduce any model and metric', icon="⚠️")

  return final_dict

def CNN(opt, path):
  recommended_images = []

  if 'feats' + path["pkl"] not in st.session_state:
    # Read dictionary pkl file
    with open(p.dict_path + '/' + path["pkl"], 'rb') as fp:
      feats = pickle.load(fp)
    st.session_state['feats' + path["pkl"]] = feats

  # Use the dictionary stored in session state
  feats = st.session_state['feats' + path["pkl"]]

  sorted_sim_per_model = similarities(opt, path, feats)

  final_dict = fusion_method(opt["fusion_method"], sorted_sim_per_model, opt["metrics"], opt["test_models"])

  for i, key in enumerate(final_dict.keys()):
    if i < 3 and path["inside_data"] == False or path["inside_data"] == True and i != 0 and i < 4:
      recommended_images.append(key)

  return recommended_images

class testDataset(Dataset): #different from train dataset, because the data organized in submission.csv is different from train.csv
   
    def __init__(self,input_image_path,other_images_path,transform=None):
      self.transform = transform
      self.input_image_path = input_image_path
      self.other_images_path = other_images_path
       
    def __getitem__(self,index):
      img0_path = self.input_image_path
      img1_path = self.other_images_path[index]
      
      img0 = Image.open(img0_path).convert('RGB')
      img1 = Image.open(img1_path).convert('RGB')

      if self.transform is not None:
        img0 = self.transform(img0)
        img1 = self.transform(img1)
      
      return img0, img1, img0_path, img1_path

    def __len__(self):
      return len(self.other_images_path)

class SiameseNetwork(nn.Module):# A simple implementation of siamese network, ResNet50 is used, and then connected by three fc layer.
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #self.cnn1 = models.resnet50(pretrained=True)#resnet50 doesn't work, might because pretrained model recognize all faces as the same.
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.2),
        )
        self.fc1 = nn.Linear(2*32*100*100, 500)
        #self.fc1 = nn.Linear(2*1000, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2)


    def forward(self, input1, input2):#did not know how to let two resnet share the same param.
        output1 = self.cnn1(input1)
        output1 = output1.view(output1.size()[0], -1)#make it suitable for fc layer.
        output2 = self.cnn1(input2)
        output2 = output2.view(output2.size()[0], -1)
        
        output = torch.cat((output1, output2),1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

def Siamese_Network(path):

  IMG_SIZE=100

  model=SiameseNetwork().cuda()
  model.load_state_dict(torch.load('/teamspace/studios/this_studio/ProjectIII/data/faces/saved_model.pth'))
  model.eval()

  folder_path = path['path']
  # Create list with all the images paths
  image_paths = glob(os.path.join(folder_path, "*.jpg"))
  if path["inside_data"] == False:
    input_image_path = path["ext_img_path"]
  else:
    input_image_path = path["input_path"]
  other_images_paths = image_paths

  testset = testDataset(input_image_path, other_images_paths, transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                                                                        transforms.ToTensor()
                                                                        ]))
  testloader = DataLoader(testset,
                          shuffle=False,
                          num_workers=2,
                          batch_size=1)

  predictions=[]
  similarities_scores = []
  similar_images_paths = []

  with torch.no_grad():
      for data in testloader:
        
          img0, img1, p0 , p1 = data
          img0, img1 = img0.cuda(), img1.cuda()
          outputs = model(img0,img1)
  
          x, predicted = torch.max(outputs, 1)
        
          # If the images are similar, then append the similarity score
          if predicted == 1:
              # print('Imagen similar')
              similarities_scores.append(x)
              #print(p1[0])
              similar_images_paths.append(p1[0])
  
          predictions = np.concatenate((predictions,predicted.cpu().numpy()),0)
  
  # Sort the images by its
  score_image_pairs = list(zip(similarities_scores, similar_images_paths))
  sorted_pairs = sorted(score_image_pairs, key=lambda x: x[0], reverse=True)

  top_n_scores, top_n_images_paths = zip(*sorted_pairs)

  top_n_images_paths=list(top_n_images_paths)

  return top_n_images_paths[:3]

def display(background, path, recommended_images):
  size = 224

  dress_title, rec1_title, rec2_title, rec3_title = st.columns(4) 
  dress, rec1, rec2, rec3 = st.columns(4) 


  if path["inside_data"] == False:
    image = Image.open(path["ext_img_path"])
  else:
    image = Image.open(path["input_path"])


  image = image.resize((size, size))
  dress_title.subheader("Your Dress")  # Add title or description 
  dress.image(image, width=size)  # Adjust width as needed

  for i, recommended_image in enumerate(recommended_images):
    rec = [dress, rec1, rec2, rec3][i+1]
    rec_title = [dress_title, rec1_title, rec2_title, rec3_title][i+1]
    try:
      # Attempt to open the recommended image (handle potential errors)
      rec_image = Image.open(recommended_image)
      rec_image = rec_image.resize((size, size))
      rec_title.subheader(f"Recommendation {i+1}")  # Add title or description 
      rec.image(rec_image, width=size)  # Adjust width as needed

    except FileNotFoundError:
      rec.write(f"Recommended Image {i+1} Not Found")

    
st.title("Image Recommendation System")

tab = 0
opt = sidebar()

if tab == 2:
  remove_background = False
else:
  remove_background = opt["background"]

path = upload(opt["background"])
# Create tabs
tabs1, tabs2 = st.tabs(['CNN', 'Siamese Networks'])

# Define content for each tab
with tabs1:
  if path is not None:
    tab = 1
    recommendations = CNN(opt, path)
    display(remove_background, path, recommendations)

# Define content for each tab
with tabs2:
  if path is not None:
    tab = 2
    recommendations = Siamese_Network(path)
    display(remove_background, path, recommendations)