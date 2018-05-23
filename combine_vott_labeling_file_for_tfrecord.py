# coding: utf-8
# author: jiankaiwang (https://jiankaiwang.no-ip.biz/)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import codecs
import sys
import math
from PIL import Image
from shutil import copyfile

# global variables
id_increament = 0
output_labels_index = 0
output_labels = {\
    "frames": {} \
    , "inputTags": "" \
    , "visitedFrames": [] \
    , "visitedFrameNames": [] \
    , "framerate": "1"\
    , "suggestiontype": "track"\
    , "scd": "false" \
}

labeldir = os.path.join('.','tmp')
labelimgdir = os.path.join('.','tmp','input')
outputdir = os.path.join('.','tmp') 
outputimgdir = os.path.join('.','tmp','output')
COMP_WIDTH = "auto"
COMP_HEIGHT = "auto"

# compressed=True: the max width is image width, not scaling
def scaleImageSize(set_width, set_height, img_width, img_height, compressed=True):    
    if str(set_width) != "auto" and str(set_height) != "auto":
        return int(set_width), int(set_height)
    elif str(set_width) != "auto" and str(set_height) == "auto":
        # set_width is not auto
        set_height = math.ceil(float(set_width)*float(img_height) / float(img_width))
        if set_height > img_height and compressed:
            set_height = img_height
    elif str(set_width) == "auto" and str(set_height) != "auto":
        set_width = math.ceil(float(set_height)*float(img_width) / float(img_height))
        if set_width > img_width and compressed:
            set_width = img_width
    else:
        set_width = img_width
        set_height = img_height
    return int(set_width), int(set_height)
    
def comporessImage(fromFile, toFile, width, height):  
    try:
        # cv2 might encounter orientation issue
        # pillow does not
        im = Image.open(fromFile)
        (img_width, img_height) = im.size
        new_width, new_height = scaleImageSize(width, height, img_width, img_height)
        (im.resize((new_width, new_height), Image.BILINEAR)).save(toFile)
        return True, float(new_height/img_height), img_width, img_height
    except Exception as e:
        print("Error: Parsing image {} is error.".format(fromFile))
        print("Error: Message is {}.".format(e))
        # directly copy the image with the same size
        copyfile(fromFile, toFile)
        return False, 1.0, 1.0, 1.0      

def read_label_file(file):
    tmpContent = ""
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            tmpContent = tmpContent + line.strip()
    return json.loads(tmpContent)

def find_all_label_files(labeldir):
    return next(os.walk(labeldir))[2]

def find_all_image_dirs(labelimgdir):
    return next(os.walk(labelimgdir))[1]

def __addFrames(jsonData):
    global output_labels_index, id_increament, order_header, COMP_WIDTH, COMP_HEIGHT
    global labelimgdir, outputimgdir
    tmpMaxId = 0
    allVisitedFrames = jsonData["visitedFrames"]
    allFrameNames = jsonData["visitedFrameNames"]
    for frameIndex in allVisitedFrames:
        if str(frameIndex) not in jsonData["frames"].keys():
            print("Warning: (no labeling) lost frames index {} on image {}.".\
                  format(str(frameIndex), jsonData["visitedFrameNames"][frameIndex]))
            continue
        output_labels["frames"][str(output_labels_index)] = jsonData["frames"][str(frameIndex)]
        
        allLabels = output_labels["frames"][str(output_labels_index)]
        for idx in range(0, len(allLabels), 1):
            if allLabels[idx]["id"] > tmpMaxId:
                tmpMaxId = allLabels[idx]["id"]
            output_labels["frames"][str(output_labels_index)][idx]["id"] += id_increament

        output_labels["visitedFrames"].append(output_labels_index)
        
        # rename the file with a index on the header of its filename
        to_dir_img = "{}_{}".format(order_header, allFrameNames[frameIndex])
        if os.path.isfile(os.path.join(labelimgdir, allFrameNames[frameIndex])):
            status, scaled_ratio, img_width, img_height = comporessImage(\
                os.path.join(labelimgdir, allFrameNames[frameIndex]) \
                , os.path.join(outputimgdir, to_dir_img) \
                , COMP_WIDTH, COMP_HEIGHT)
            if not status:
                print("Error: Can not compress the image {}."\
                      .format(allFrameNames[frameIndex]))
        else:
            print("Error: Lose image {}".format(os.path.join(labelimgdir, allFrameNames[frameIndex])))
        output_labels["visitedFrameNames"].append(to_dir_img)   
        output_labels_index = output_labels_index + 1
        
    # get max id and add one, 
    # it is the beginning index of the next labelling files
    id_increament += tmpMaxId + 1

def __addTags(jsonData):
    global output_labels
    if len(output_labels['inputTags']) > 1:
        currentTags = output_labels['inputTags'].strip().split(',')
    else:
        currentTags = []
    inputTags = jsonData['inputTags'].strip().split(',')
    for tag in inputTags:
        if tag not in currentTags:
            currentTags.append(tag)
    output_labels['inputTags'] = ','.join(currentTags)

def add_label_file(file):
    labelData = read_label_file(file)      
    __addFrames(labelData)
    __addTags(labelData)
    
def output_all_labels(outputdir):
    global output_labels
    outputPath = os.path.join(outputdir, 'output.json')
    with codecs.open(outputPath, 'w', 'utf-8') as fout:
        fout.write(json.dumps(output_labels, ensure_ascii=True))
        
def __check_input_output():
    global labeldir, labelimgdir, outputdir, outputimgdir
    
    # check image pooling directory
    if (not os.path.isdir(labeldir)) or (not os.path.isdir(labelimgdir)):
        print("Error: no labelling folder {} or no image input folder {}"\
              .format(labeldir, labelimgdir))
        return False  
    
    # check output file path
    if not os.path.isdir(outputdir):
        try:
            os.mkdir(outputdir)
        except:
            print("Error: No such labelling output directory and can not create it.")
            return False  
            
    if not os.path.isdir(outputimgdir):
        try:
            os.mkdir(outputimgdir)  
        except:
            print("Error: No such image output directory and can not create it.")
            return False
            
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument("--labeldir" \
                        , type=str \
                        , default=os.path.join('.','tmp') \
                        , help="dir conserves label files")      
    parser.add_argument("--labelImgdir" \
                        , type=str \
                        , default=os.path.join('.','tmp','input') \
                        , help="dir conserves images mapping to labelling files")       
    parser.add_argument("--outputdir" \
                        , type=str \
                        , default=os.path.join('.','tmp') \
                        , help="combined label file")      
    parser.add_argument("--outputImgdir" \
                        , type=str \
                        , default=os.path.join('.','tmp','output') \
                        , help="combined label file") 
    parser.add_argument("--compressedwidth" \
                        , type=str \
                        , default="auto" \
                        , help="compress the image by width")      
    parser.add_argument("--compressedheight" \
                        , type=str \
                        , default="auto" \
                        , help="compress the image by height")      
    args = parser.parse_args()
    
    is_available = False
    
    # get args value
    labeldir = args.labeldir
    labelimgdir = args.labelImgdir
    outputdir = args.outputdir  
    outputimgdir = args.outputImgdir 
    COMP_WIDTH = args.compressedwidth
    COMP_HEIGHT = args.compressedheight
    
    # others variable
    # due to vott using the order related to the directory
    # you have to rename the file to avoid the misordering
    # while combining all vott labelling files
    order_header = 0

    if not __check_input_output():
        print("Error: Can not validate input {}, {} or output {} {}."\
              .format(labeldir, labelimgdir, outputdir, outputimgdir))
        sys.exit(1)
        
    # parse directory
    allLabelFiles = find_all_label_files(labeldir)
    for files in allLabelFiles:
        order_header += 1
        print('Parse file {}.'.format(files))
        add_label_file(os.path.join(labeldir, files))
    output_all_labels(outputdir)
    
    
    
    
    
    