import requests
import zipfile
import os
from PIL import Image
#for splitting
from sklearn.model_selection import train_test_split
#for tensorflow
import tensorflow as tf

def crop(image):
    breadth, length = image.size
    #cropped and centered
    new_b = min(breadth, length)
    new_l = new_b

    left = (breadth- new_b)/2
    right = (breadth+new_b)/2
    top = (length - new_l)/2
    bottom = (length + new_l)/2


    return image.crop((left,top,right,bottom))

def standardise(image):
    rgb_img = Image.new("RGB", image.size)
    rgb_img.paste(image)
    return rgb_img

#extract images
split_images = []
split_labels = []
path = "/Users/j.harini/Downloads/dataset_.zip"
with zipfile.ZipFile(path, "r") as rock_files:
    rock_files.extractall("extracted_images")
    
#standardise/process images
    for (root,_,files) in os.walk("extracted_images"):
        for f in files:
            img_path = os.path.join(root, f)
           
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"Skipping non-image file: {f}") 
                continue
            
            try:
                img = Image.open(img_path)
                img.load()
                img = crop(img)
                img = img.resize((200,200))
                img = standardise(img)
            except Exception as e:
                print(f"Skipping corrupted image: {img_path} due to error: {e}")
                continue

            path = img_path.split(os.path.sep)
            rock_type = path[-3]
            rock_name = path[-2]
            label = f"{rock_type}_{rock_name}"
            split_images.append(img_path)
            split_labels.append(label)

print("Total images collected:", len(split_images)) #debug
print("Sample image paths:", split_images[:5]) #debug

print(f"Total images found: {len(split_images)}") #debug
print("Labels: ", set(split_labels)) #debug

#Splitting into training and testing sets
if len(split_images) > 0:
    train_images, test_images, train_labels, test_labels = train_test_split(split_images, split_labels, test_size = 0.2, random_state = 42) #making sure the same set of images split everytime with random state = 42.
else:
    print("No images to split.") #must check dataset if this occurs.


#Pre-Trained Model with Tensorflow
img_size = (200, 200) #same size as the images we have 
batch_size = 32
classes = len(set(split_labels))

base = MobileNetV2(input_shape = (200,200,3), include_top = False) #no top layer.
base.trainable = False #all other layers frozen




