from PIL import Image
import numpy as np

def process_image(inp):
    img=Image.open(inp).convert('RGB')
    img=img.resize((224,224))
    img_array=np.array(img,dtype=np.float32)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
