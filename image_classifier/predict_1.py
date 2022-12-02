
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from operator import itemgetter
import data_inputs as d
import json
import argparse

image_shape_normalized  = 224
json_path="label_map.json"

def return_model():
    # import module
    load_keras_model = tf.keras.models.load_model(saved_model_filepath,custom_objects={'KerasLayer':hub.KerasLayer})
    load_keras_model.summary()
    return load_keras_model

#import class_names
def load_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)

    return class_names
    
#Process Image
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_shape_normalized, image_shape_normalized))
    image /= 255
    return image

mapped_label={}

def predict_output(image_path, model, top_k,json_path):
    #Open image
    open_im = Image.open(image_path)
    converted_image = np.asarray(open_im)
    
    #process image
    processed_test_image = process_image(converted_image)
    test_image  = np.expand_dims(processed_test_image, axis=0)  #expand dimension
    
    #predict probability
    ps = model.predict(test_image)
    ps_list=list(ps[0])
    
    #map class names and probability into one list
    for label,i in zip(d.dataset_info.features['label'].names,ps_list):
        mapped_label[label]=i
        
    #get top k modules
    top_k_labels=dict(sorted(mapped_label.items(), key = itemgetter(1), reverse = True)[:top_k])
    
    top_classes=list(top_k_labels.keys())
    classes=[]
    class_names=load_class_names(json_path)
    for i in class_names:
        if class_names[i]  in top_classes:
            classes.append(i)
         
    return classes

#import path
def output(top_k,json_path):
    #path_list=glob.glob("./test_images/*.jpg")

    labels = predict_output(image_path,return_model(),top_k,json_path)

    names=[]
    result_ps={}
    class_names=load_class_names(json_path)
    for i in labels:
        names.append(class_names[i])
        result_ps[class_names[i]]=mapped_label[class_names[i]]

    #get dictonary with highest value
    max_value = max(result_ps, key=result_ps.get)
    print(result_ps)
    print(f"\n Name of the flower is  {max_value}")
    print("\n The Probability is {0}".format(result_ps[max_value]))

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("model")
parser.add_argument('--top_k', default=5,type=int,help="List of Top K classes")
parser.add_argument("-p",'--category_names', default="label_map.json",type=str,help="Map classes based on json file")
args=parser.parse_args()

image_path=args.image
saved_model_filepath=args.model
json_path=args.category_names

if args.top_k:
    top_classes=predict_output(image_path,return_model(),args.top_k,json_path)
    print("Top class")
    print(top_classes)
elif args.category_names:
    json_path=args.category_names
    output(args.top_k,json_path)

output(args.top_k,json_path)

"""
Default Top_k = 5,category_names=label_map.json

give the following argument

basic usage
py -m predict_1 "./test_images_2/fire lily.jpg" trained_model.h5

options
py -m predict_1 "./test_images_2/fire lily.jpg" trained_model.h5 --top_k 2


"""