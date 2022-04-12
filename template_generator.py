from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import os

def main():

    generate_code128_templates()

def generate_code128_templates():

    font_path = "./fonts/code128.ttf"     
    font = ImageFont.truetype(font_path, 96)
    
    image = np.zeros((100,50),np.uint8)
    
    #Check if dir exists
    if(not os.path.isdir("./Code128CharSet")):
        os.mkdir("./Code128CharSet")
    
    #Generate characters
    for i in range(32, 126):
        #Make image have white background
        image[image >= 0] = 255
        pil_image = Image.fromarray(image)
        canvas = ImageDraw.Draw(pil_image)
        canvas.text((0, 0),  chr(i), font = font)
        image = np.array(pil_image)
        path = os.path.join("Code128CharSet", "Code128_" + str(i) + ".png")
        
        variable_crop(path, image, font.getsize(chr(i))[0])
        
    #Generate special characters
    for i in list(range(195, 206)) + list(range(207,208)):
        #Invert Image
        image[image >= 0] = 255
        pil_image = Image.fromarray(image)
        canvas = ImageDraw.Draw(pil_image)
        canvas.text((0, 0),  chr(i), font = font)
        image = np.array(pil_image)
        path = os.path.join("Code128CharSet", "Code128_" + str(i) + ".png")
        
        variable_crop(path, image, font.getsize(chr(i))[0])
        
    #Generate stop character
    image[image >= 0] = 255
    pil_image = Image.fromarray(image)
    canvas = ImageDraw.Draw(pil_image)
    canvas.text((0, 0),  chr(206), font = font)
    image = np.array(pil_image)
    path = os.path.join("Code128CharSet", "Code128_" + str(206) + ".png")
    variable_crop(path, image, font.getsize(chr(206))[0], False)    
           
def variable_crop(path, image, char_width, crop_font = True):
    temp = image.copy()
    
    #Threshold image to inverse colours
    temp = 255*(temp <= 127).astype(np.uint8)
    
    #Get bounding box of region of interest
    points = cv2.findNonZero(temp)
    x, y, w, h = cv2.boundingRect(points)
    
    if(crop_font is True):
        rect = image[y:y+h, x:x+char_width]
    else:
        rect = image[y:y+h, x:x+w]
    cv2.imwrite(path, rect)

if __name__ == '__main__':
    main()