import os
import cv2  
import glob
import random
import traceback
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import tensorflow as tf


def load_model(model_name, model_load_unable_error, weights='imagenet'):
  try:
    if model_name.lower() == "vgg16":
      model = tf.keras.applications.VGG16(include_top=False, weights=weights)
      preprocess_image = tf.keras.applications.vgg16.preprocess_input

    elif model_name.lower() == "vgg19":
      model = tf.keras.applications.VGG19(include_top=False, weights=weights)
      preprocess_image = tf.keras.applications.vgg16.preprocess_input

  except Exception as e:
    model_load_unable_error.text("Model Not Found")
  model.trainable = False
  conv_layers = [model_layer.name for model_layer in model.layers if isinstance(model_layer, tf.keras.layers.Conv2D)]
  return {"model" : model, "conv_layers":conv_layers, "preprocess_image": preprocess_image}

def get_layer_weights(conv_layers, chosen_layers, layer_type):
  chart_placeholder = st.sidebar.empty()
  max_layer_weight = 1.0
  layer_weight_func = lambda layer_name, chosen_layers : st.sidebar.slider(
      label=layer_name, min_value=0.0, max_value=max_layer_weight,
      value=1.0 if layer_name in chosen_layers else 0.0,
      step=0.05, key='slider_'+layer_name+'and'+layer_type)

  data = pd.DataFrame.from_records(columns=['layer', 'weight'],
    data=[(name, layer_weight_func(name, chosen_layers)) for name in conv_layers])
  
  layer_weights = { row['layer'] : row['weight'] for index, row in data.iterrows() 
                    if row['weight'] > 0 }
  return layer_weights

def reshape_image(image, size):
  image_prep = tf.image.resize(image, size=size, method='lanczos5') 
  image_prep = image_prep[tf.newaxis, ..., :3]
  return image_prep

def compute_gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)



def build_content_layer_map(features, content_layers):
  content_map = { layer_name : features[layer_name] for layer_name in content_layers }
  return content_map

def build_style_layer_map(features, style_layers):
  gram_norm = lambda feature_layer_name: compute_gram_matrix(feature_layer_name) 
  style_map = { layer_name : gram_norm(features[layer_name])
                  for layer_name in style_layers
                }
  return style_map


def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]
  return x_var, y_var


def style_content_loss( output_content_map,output_style_map, content_layer_weights,style_layer_weights, content_targets, \
  style_targets,content_reconstruction_weight, style_reconstruction_weight, total_variation_weight,output_image):
  content_loss = tf.add_n([content_layer_weight * tf.reduce_mean(
                              (output_content_map[content_layer_name] - content_targets[content_layer_name])**2) 
                              for content_layer_name, content_layer_weight in content_layer_weights.items()
                              if content_layer_weight > 0 ]) 

  style_loss = tf.add_n([style_layer_weight * tf.reduce_mean(
                            (output_style_map[style_layer_name] - style_targets[style_layer_name])**2 ) 
                            for style_layer_name, style_layer_weight in style_layer_weights.items()
                            if style_layer_weight > 0]) 
  total_loss = content_reconstruction_weight*content_loss + style_reconstruction_weight*style_loss 
  if total_variation_weight > 0:
    x_var, y_var = high_pass_x_y(output_image)
    variation_loss = tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))
    total_loss += total_variation_weight * variation_loss
  return total_loss

def clipping(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

class NeuralStyleTransfer:
  def __init__(self):
    tf.keras.backend.clear_session()

  def __call__(self, content_img, style_img, 
              epochs, size, content_layer_weights, style_layer_weights, 
              content_reconstruction_weight, style_reconstruction_weight, total_variation_weight, optimizer):

    self.content_reconstruction_weight = content_reconstruction_weight
    self.style_reconstruction_weight = style_reconstruction_weight
    self.total_variation_weight = total_variation_weight
    self.content_layer_weights = content_layer_weights
    self.style_layer_weights = style_layer_weights
    self.optimizer = optimizer

    content_resized = reshape_image(content_img, (size, size))
    style_resized = reshape_image(style_img, (size, size))

    content_prep = self.preprocess_image(content_resized)
    style_prep = self.preprocess_image(style_resized)

    layers_of_interest = list(set(content_layer_weights).union(style_layer_weights))
    outputs = {layer_name : self.model.get_layer(layer_name).output for layer_name in layers_of_interest}

    self.feature_extractor = tf.keras.Model(self.model.inputs, outputs)
    input_content_features = self.feature_extractor(tf.constant(content_prep)) 
    input_style_features = self.feature_extractor(tf.constant(style_prep))
    self.content_targets = build_content_layer_map(input_content_features, content_layer_weights.keys())
    self.style_targets = build_style_layer_map(input_style_features, style_layer_weights.keys())
    output_image = tf.Variable(content_resized)


    for epoch in range(epochs):
      self.train_step(output_image)
      output_image_resized = tf.image.resize(output_image[0], size=content_img.shape[:2], method='bilinear')
      output_img_array = np.array(output_image_resized, np.uint8)
      yield epoch + 1, output_img_array



  @tf.function 
  def train_step(self, output_image):
    with tf.GradientTape() as tape:
      output_prep = self.preprocess_image(output_image.value())
      output_features = self.feature_extractor(output_prep)
      
      output_content_map = build_content_layer_map(output_features, content_layer_weights.keys())
      output_style_map = build_style_layer_map(output_features, style_layer_weights.keys())
      total_loss = style_content_loss( output_content_map,output_style_map,self.content_layer_weights, self.style_layer_weights, self.content_targets, \
  self.style_targets,self.content_reconstruction_weight, self.style_reconstruction_weight, self.total_variation_weight, output_image)

    grads = tape.gradient(total_loss, output_image)  
    optimizer.apply_gradients([(grads, output_image)])
    output_image.assign(clipping(output_image))

try:
  st.title("Neural Style Transfer With Tensorflow")
  progress_text = st.empty()
  output_image_placeholder = st.empty()
  model_load_unable_error = st.empty()
  progress_text.text("Preparing given data for applying NST ...")
  progress_bar = st.progress(0)

  st.sidebar.text("Neural Style Transfer")
  style_image = cv2.imread("./images/logo_image/nst_image.jpg")
  st.sidebar.image(image=np.asarray(style_image), use_column_width=True, 
    caption='', clamp=True, channels='RGB')


  video_required = st.sidebar.number_input(label='Is video of output required(1 for yes , 0  for No)', min_value=0, max_value=1, value=0, step=1)
  images_required = st.sidebar.number_input(label='Are images of output required(1 for yes , 0  for No)', min_value=0, max_value=1, value=0, step=1)

  st.sidebar.header("Set images and various parameters ")
  content_path = st.sidebar.selectbox('Please Select Content Image', options=glob.glob('images/content_images/*.*'),
                                      format_func=lambda glob_path: os.path.basename(glob_path), index = 0, key = "content_path")
  content_image = cv2.imread(content_path)
  st.sidebar.image(image=np.asarray(content_image), use_column_width=True, 
    caption='content image', clamp=True, channels='RGB')
  style_path = st.sidebar.selectbox('Please Select Style Image', options=glob.glob('images/style_images/*.*'),
                                    format_func=lambda glob_path: os.path.basename(glob_path),index=0,  key = "style_path")
  style_image = cv2.imread(style_path)
  st.sidebar.image(image=np.asarray(style_image), use_column_width=True, 
    caption='style image', clamp=True, channels='RGB')


  epochs = st.sidebar.slider(label='Number of Epochs', min_value=1, max_value=512, value=16, key = 'epochs')
  model_name = st.sidebar.selectbox(label='Select Model to Apply',  options=['VGG16', 'VGG19'], index=0,  key = "model_name")
  size = st.sidebar.slider(label='Image Size', min_value=128, max_value=1024, value=512, step=1, format='%d', key="size")
  content_weight = st.sidebar.slider(label='Content Weight', min_value=32, max_value=2048, value=64, key = 'content_weight')
  style_weight = st.sidebar.slider(label='Style  weight', min_value=128, max_value=2048, value=256, key = 'style_weight')
  total_weight = st.sidebar.slider(label='Total Variation Weight', min_value=0, max_value=256, value=16, key = 'total_weight')


  nst = NeuralStyleTransfer()
  model_dict = load_model(model_name, model_load_unable_error) 
  nst.model = model_dict["model"]
  nst.preprocess_image = model_dict["preprocess_image"]
  conv_layers = model_dict["conv_layers"]
  assert len(conv_layers) > 0, "This model does not contains any convolutional layers."
  st.sidebar.subheader("Content layer weights")
  content_layer_weights = get_layer_weights(conv_layers, conv_layers[:1], 'content')
  st.sidebar.subheader("Style layer weights")
  style_layer_weights = get_layer_weights(conv_layers, conv_layers[:1], 'style')
  assert len(content_layer_weights) > 0 and len(style_layer_weights) > 0, \
    f"At least one either content or style layer must have a weight greater than 0"


  loss_function = st.sidebar.selectbox(label='Select Loss Function',  options=['Adam'], index=0,  key = "model_name")
  learning_rate = st.sidebar.slider(label='Set Learning Rate', min_value=0.0, max_value=1.0, value=0.8, key = 'learning_rate')
  epsilon = st.sidebar.slider(label='Set epsilon', min_value=0.0, max_value=1.0, value=0.1, key = 'epsilon')





  if loss_function.lower() == "adam":
    optimizer = tf.optimizers.Adam(learning_rate,epsilon)

  # print(content_path.split("\\")[1])
  content_file_name =  content_path.split("\\")[1].split(".")[0]


  if images_required ==1 or video_required ==1:
    if not os.path.exists("./outputs/"):
      os.mkdir("./outputs/")

    if video_required == 1:
      if not os.path.exists("./outputs/videos/"):
        os.mkdir("./outputs/videos")
      video = cv2.VideoWriter(f"./outputs/videos/{content_file_name}.avi", 0, 1, (content_image.shape[1],content_image.shape[0]))

    if images_required ==1:
      if not os.path.exists("./outputs/images/"):
        os.mkdir("./outputs/images")

  def write_video_and_audio( epoch,  output_image, content_file_name,content_image,  video_required = 0,images_required = 0):

    if video_required == 1:
      global video
      video.write(output_image)

    if images_required ==1:
      cv2.imwrite(f'./outputs/images/color_img{epoch}.jpg ', output_image)
    return


  for epoch, output_image in nst(content_image, style_image, epochs, size, 
                                          content_layer_weights, style_layer_weights, 
                                          content_weight, style_weight, 
                                          total_weight, optimizer):

    progress_text.text(f"Epoch {epoch}/{epochs}")

    progress_bar.progress(epoch/epochs)
    
    op= cv2.normalize(output_image, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    output_image_placeholder.image(output_image, caption='Styled Image', use_column_width=True, clamp=True, channels='RGB')

    if images_required ==1 or video_required ==1:
      write_video_and_audio(epoch,output_image, content_file_name,content_image, video_required,images_required)
  progress_text.text(f"NST for {epochs} epochs Completed!")
  progress_bar.empty()

  if video_required==1:
    video.release()

  cv2.destroyAllWindows() 
except Exception as e:
  print("SOME PROBLEM OCCURED")
