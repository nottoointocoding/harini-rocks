import requests
import zipfile
import io #converts raw bytes from rock_list[i] into a valid file path, i.e, an image object.
import os
from PIL import Image
from sklearn.model_selection import train_test_split
'''#download zip file
zip_link = "https://www.dropbox.com/scl/fi/8kftgh4w347sokf8ub0tg/dataset_.zip?rlkey=ewpgq6qht0azmz053pf2ssbfw&st=pfyeayvf&dl=1"

download = requests.get(zip_link)

with open("rock_files.zip", "wb") as f:
    f.write(download.content)  '''

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
path = "/Users/j.harini/Downloads/dataset_.zip"
with zipfile.ZipFile(path, "r") as rock_files:
    rock_files.extractall("extracted_images")
    rock_list = rock_files.namelist() #gives us all file names: used for img_path
    '''f = rock_files.read(rock_list[0])
    print(f[:10])'''
#standardise/process images
    for i in range(5):
        f = io.BytesIO(rock_files.read(rock_list[i])) #image object instead of raw bytes
        img_path = os.path.join("extracted_images", rock_list[i]) #extract path is where files will be saved
        img = Image.open(f)
        img.load()
        img = crop(img)
        img = img.resize((200,200))
        img = standardise(img)
        img.show()

#Splitting into training and testing sets
        
        
        




