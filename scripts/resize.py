from PIL import Image

img = Image.open("../data/test/origin.png").convert("RGB").resize((192, 64)) 
img.save('resized_image.png')