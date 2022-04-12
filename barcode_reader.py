import cv2
import numpy as np
import math
from scipy import ndimage
import argparse
import os

arg_parse = argparse.ArgumentParser(description='List the content of a folder')
arg_parse.add_argument('path',type=str, help='Path to image')

def main():

    args = arg_parse.parse_args()
    
    if(os.path.isfile(args.path)):
        frame = cv2.imread(args.path)
    else:
        print("File not found. Scan has failed")
        return
    
    scanning_ended = False
    
    while(not scanning_ended):

        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Canny Edge Detection and binaraize image within threshold
        canny_edges = cv2.Canny(grey_frame,127,220)
        
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        morphed_frame = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, morph_kernel)
        
        morphed_frame = cv2.erode(morphed_frame, None, iterations=10)
        morphed_frame = cv2.dilate(morphed_frame, None, iterations=10)
        
        contours, hierarchy = cv2.findContours(morphed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if(contours is None or len(contours) == 0):
            continue
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        
        mask = np.zeros(frame.shape, dtype='uint8')
        cv2.drawContours(mask,[box_points],0,(255,255,255),2)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        lines = cv2.HoughLines(mask, 1, np.pi/180, 255)
        #No lines detected, image too small resize mask
        if(lines is None):
            for image_scale in np.arange(1.0, 2.0, 0.1):
                image_scale = round(image_scale, 2)
                scaledFrame = cv2.resize(mask, (int(mask.shape[1]*image_scale), int(mask.shape[0]*image_scale)), interpolation=cv2.INTER_CUBIC)
                lines = cv2.HoughLines(scaledFrame, 1, np.pi/180, 255)
                if(lines is not None):
                    break
            if(lines is None):
                print("Hough lines could not be detected. Scan has failed")
                return
            
        #Get longest line to determine barcode orientation relative to horizontal.
        longest_line_magnitude = 0.0
        longest_line_angle = 0.0
        
        #Find Hough lines
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                pt_x0 = a*rho
                pt_y0 = b*rho
                pt_x1 = int(pt_x0 + 1000*(-b))
                pt_y1 = int(pt_y0 + 1000*(a))
                pt_x2 = int(pt_x0 - 1000*(-b))
                pt_y2 = int(pt_y0 - 1000*(a))
                
                if(longest_line_magnitude < math.sqrt(math.pow(pt_x2-pt_x1, 2) + math.pow(pt_y2-pt_y1, 2))):
                    longest_line_angle = math.degrees(theta)

                cv2.line(mask,(pt_x1,pt_y1),(pt_x1,pt_y2),(0,0,255),1)
            
            #Orient Horizontally
            if(longest_line_angle > 0 and longest_line_angle <= 90):
                rotate_angle = longest_line_angle - 90
            elif(longest_line_angle <= 180 and longest_line_angle > 90):
                rotate_angle = longest_line_angle + 90
            elif(longest_line_angle <= 270 and longest_line_angle > 180):
                rotate_angle = longest_line_angle - 90
            elif(longest_line_angle == 0 or longest_line_angle > 270):
                rotate_angle = longest_line_angle - 90
            
            stabilized_image = ndimage.rotate(frame, rotate_angle)
            grey_frame = cv2.cvtColor(stabilized_image, cv2.COLOR_BGR2GRAY)
            
            #Perform previous steps with horizontal image for higher accuracy instead of rotating rect
            
            #Canny Edge Detection
            canny_edges = cv2.Canny(grey_frame,215,220)
            
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            morphed_frame = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, morph_kernel)

            morphed_frame = cv2.erode(morphed_frame, None, iterations=5)
            morphed_frame = cv2.dilate(morphed_frame, None, iterations=5)
            
            
            contours, _ = cv2.findContours(morphed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if(contours is None or len(contours) == 0):
                continue
                
            #Sort contours from largest area to smallest. Grabs largest one
            c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            rect = cv2.minAreaRect(c)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            
            min_x, min_y, _ = stabilized_image.shape
            max_x, max_y = 0, 0
            
            for i in box_points:
                if(i[0] < min_x):
                    min_x = i[0]
                if(i[1] < min_y):
                    min_y = i[1]
                    
                if(i[0] > max_x):
                    max_x = i[0]
                if(i[1] > max_y):
                    max_y = i[1]
                    
            x = min_x
            y = min_y
            width = max_x - min_x
            height = max_y - min_y
            
            #Crop out part of image
            crop_image = stabilized_image[int(y):int(y+height), int(x):int(x+width)]
            print("Decoding")
            if(decode_code128_barcode(crop_image) is True):
                scanning_ended = True
            else:
            
                #Attempt multi-scale template matching
                for image_scale in np.arange(0.2, 2.0, 0.1):
                    image_scale = round(image_scale, 2)
                    scaledFrame = cv2.resize(crop_image, (int(crop_image.shape[0]*image_scale), int(crop_image.shape[1]*image_scale)), interpolation=cv2.INTER_CUBIC)
                    print("Decoding at scale: " + str(image_scale))
                    if(decode_code128_barcode(scaledFrame) is True):
                        scanning_ended = True
                        break
                
                if(scanning_ended is True):
                    break
                
                #Check if upside down
                print("Flipping Image")
                crop_image = ndimage.rotate(crop_image, 180)
                if(decode_code128_barcode(crop_image) is True):
                    scanning_ended = True
                else:
                    for image_scale in np.arange(0.2, 2.0, 0.1):
                        image_scale = round(image_scale, 2)
                        scaledFrame = cv2.resize(crop_image, (int(crop_image.shape[0]*image_scale), int(crop_image.shape[1]*image_scale)), interpolation=cv2.INTER_CUBIC)
                        print("Decoding at scale: " + str(image_scale))
                        if(decode_code128_barcode(scaledFrame) is True):
                            break
                        scanning_ended = True
        
        cv2.waitKey(1)
    # Destroy all the windows
    cv2.destroyAllWindows()
    
    
def decode_code128_barcode(barcode):
    
    barcode_string = ""

    bar_height, bar_width, _ = barcode.shape
    
    start_x = 0
    end_x = 0
    region = 1
    
    while(True):
        
        end_x = find_end_x(start_x, barcode)
        
        if(start_x >= bar_width):
            break
        
        region_of_interest = barcode[0:bar_height , start_x:end_x]
        
        #Max correlation coefficent and occurrence location
        max_correlation_and_value = [0,0]
        for i in list(range(32, 126)) + list(range(195, 207)):
    
            template = cv2.imread('./Code128CharSet/Code128_' + str(i) + '.png')
            template = cv2.resize(template, (end_x-start_x+1, bar_height))
            
            matched = cv2.matchTemplate(region_of_interest, template, cv2.TM_CCORR_NORMED)
            
            #Get maximum correlation
            _, max_val, _, _ = cv2.minMaxLoc(matched)

            #print(chr(i) + " has a correlation match score: " + str(round(max_val*100, 2)) +"% for region " + str(region))
            
            if(max_val > max_correlation_and_value[0]):
                max_correlation_and_value[0] = max_val
                max_correlation_and_value[1] = i

        barcode_string = barcode_string + chr(max_correlation_and_value[1])
        start_x = end_x + 1
        region = region + 1

    #Code128 values different from ascii values
    #Checksum can be calculated as value of start character + value of data character*data position then apply modulo 103 on the sum and take the remainder
    checksum = ord(barcode_string[0]) - 32 if ord(barcode_string[0]) <= 126 else ord(barcode_string[0]) - 100
    position = 1
    
    for i in barcode_string[1:-2]:
        if(ord(i) <= 126):
            value = ord(i) - 32
        else:
            value = ord(i) - 100
        checksum = checksum + value*position
        position = position + 1
    
    checksum_char_value =  checksum % 103
    checksum_char = chr(checksum_char_value + 32) if checksum_char_value <= 126 else chr(checksum_char_value + 100)

    decoded = ""
    in_setC = False
    
    if(len(barcode_string) >= 2):
        #Check for switch character sets
        for i in range(1, len(barcode_string)-2):
            
            
            #Switch to Code C set if true
            if(barcode_string[i] in ("Ç") and in_setC is False):
                in_setC = True
            #Switch to Code A or B set
            elif(barcode_string[i] in ("È","É") and in_setC is True):
                in_setC = False
            #Set C decoding
            elif(in_setC is True):
                if(ord(barcode_string[i]) <= 126):
                    value = ord(barcode_string[i]) - 32
                else:
                    value = ord(barcode_string[i]) - 100
                decoded = decoded + str(value)
                
            else:
                decoded = decoded + barcode_string[i]     
            
    if(len(barcode_string) >= 2 and checksum_char == barcode_string[-2]):
        #Check if stop character detected successfully. Indicated bar code was fully read
        if(barcode_string[-1] in ("Î", "Ó", "û")):
        
            if(barcode_string[0] in ("Ë", "Ð", "ø", "Ì", "Ñ", "ù", "Í", "Ò", "ú")):
                print("Code 128 barcode decoded successfully. Output string in \"\": \"" + decoded + "\"")
                return True

    print("Barcode not decoded correctly, full fragment: " + barcode_string)
    return False
    
def find_end_x(start_x, image):
    #Barcode modules have 3 black bars and 3 white bars. Always start with black bar and end with white bar
    grey_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = grey_frame.shape
    black_num = 0
    on_black = False
    white_num = 0
    on_white = False
    
    for i in range(start_x, cols):
        if(grey_frame[rows//2, i] <= 127):
            if(on_black == False):
                black_num += 1
                on_black = True
                on_white = False
                
        if(grey_frame[rows//2, i] >= 128):
            if(on_white == False):
                white_num += 1
                on_black = False
                on_white = True      
                
        if(black_num == 4):
            #Look for end character if near end of image
            if((i-1) + (i-1 - start_x) >= grey_frame.shape[1]):
                return grey_frame.shape[1]
            return i-1
            
    return grey_frame.shape[1]            

if __name__ == '__main__':
    main()