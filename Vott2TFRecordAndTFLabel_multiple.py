# coding: utf-8
# author: jiankaiwang (https://jiankaiwang.no-ip.biz/)

import json
import os
import tensorflow as tf
import cv2
from object_detection.utils import dataset_util
import codecs
import math
import argparse

# global variable
labelDir = os.path.join('.','raw','origindata')
outputFilePath = os.path.join('.','data')
outputLabelFile = os.path.join(outputFilePath, 'image_label.pbtxt')
outputTrainTFRecordFile = os.path.join(outputFilePath, 'train.tfrecords')
outputEvalTFRecordFile = os.path.join(outputFilePath, 'eval.tfrecords')
TRAIN_VALIDARION_RATIO = 0.8  # TRAIN:VALIDARION = 8:2

# private variable
__allTages = []
__trainLabels = []
__evalLabels = []

class EXAMPLE:
    height = 0
    width = 0
    filename = ""
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

def create_tf_example(example, imgInByte):
    height = example.height # Image height
    width = example.width # Image width
    filename = example.filename # Filename of the image. Empty if image is not from file
    encoded_image_data = imgInByte # Encoded image bytes
    image_format = example.image_format # b'jpeg' or b'png'
    
    xmins = example.xmins # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = example.xmaxs # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = example.ymins # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = example.ymaxs # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = example.classes_text # List of string class name of bounding box (1 per box)
    classes = example.classes # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={\
    'image/height': dataset_util.int64_feature(height) \
    , 'image/width': dataset_util.int64_feature(width) \
    , 'image/filename': dataset_util.bytes_feature(filename) \
    , 'image/source_id': dataset_util.bytes_feature(filename) \
    , 'image/encoded': dataset_util.bytes_feature(encoded_image_data) \
    , 'image/format': dataset_util.bytes_feature(image_format) \
    , 'image/object/bbox/xmin': dataset_util.float_list_feature(xmins) \
    , 'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs) \
    , 'image/object/bbox/ymin': dataset_util.float_list_feature(ymins) \
    , 'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs) \
    , 'image/object/class/text': dataset_util.bytes_list_feature(classes_text) \
    , 'image/object/class/label': dataset_util.int64_list_feature(classes) \
    }))
    return tf_example

def getAllFileList(getVottPath):    
    filenames = next(os.walk(getVottPath))[2]
    return filenames

def getJsonFile(getFileName):
    tmpContent = ""
    with codecs.open(os.path.join(labelDir, getFileName), 'r') as fin:
        for line in fin:
            tmpContent += line.strip()
    return json.loads(tmpContent)
    
def conserveTagName(getTagList):
    for tags in getTagList:
        if tags not in __allTages:
            __allTages.append(tags)

def writePbtxt(outputLabelFile):
    with codecs.open(outputLabelFile, 'w', 'utf-8') as fout:
        for item in __allTages:
            fout.write("item {\r\n  id: " + str(__allTages.index(item) + 1) \
                    + '\r\n  name: \'' + item + "\'\r\n}\r\n")

def parseExampleObject(img_height, img_width, filename, fileformat\
    , xmins, xmaxs, ymins, ymaxs, classes_text, classes):
    global __allTages, labelDir
    egObj = EXAMPLE()
    egObj.height = img_height
    egObj.width = img_width
    egObj.filename = filename
    egObj.image_format = fileformat
    egObj.xmins = xmins
    egObj.xmaxs = xmaxs
    egObj.ymins = ymins
    egObj.ymaxs = ymaxs
    egObj.classes_text = classes_text
    egObj.classes = classes     
    return egObj

def getImgEncode(getPath):  
    # you have to use tf.gfile.FastGFile to encode
    # or error message: [Unable to decode bytes as JPEG, PNG, GIF, or BMP]
    image_data = tf.gfile.FastGFile(getPath, 'rb').read()
    return image_data
    
def indexTrainValidate(ttlFileCount):
    global TRAIN_VALIDARION_RATIO
    evalTtlCount = math.ceil(ttlFileCount*(1.0-TRAIN_VALIDARION_RATIO))
    return 0, ttlFileCount-evalTtlCount
    
def sepTrainValidateData(getJsonContent, example, isTrain=True):
    visitedFrameNames = getJsonContent['visitedFrameNames']

    # get trained and validated index
    train_start, eval_start = indexTrainValidate(len(visitedFrameNames))
    if not isTrain:
        train_start, eval_start = eval_start, len(visitedFrameNames)
    
    # get trained and validated index
    for frames_index in range(train_start, eval_start, 1):
        # check to have labeling data
        if str(frames_index) not in getJsonContent['frames'].keys():
            print("Error: Frames {} has no key {} in output {}.".format(\
                  visitedFrameNames[frames_index], frames_index, example))
            continue
        
        # get original image height, width, channels
        crtFileName = os.path.join(\
                    labelDir\
                    , example.split('.')[0]\
                    , visitedFrameNames[frames_index])
        
        try:
            img_height, img_width, img_channel = cv2.imread(crtFileName).shape
        except:
            print("Error: Parsing image {} is error.".format(crtFileName))
            continue

        # start to parse labeling data
        labelData = getJsonContent['frames'][str(frames_index)]
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        for eachLabeling in range(0, len(labelData), 1): 
            labelInfo = labelData[eachLabeling]
            normalized_width = float(labelInfo["width"])
            normalized_height = float(labelInfo["height"])
            # feed the normalized value into list
            x1 = float(labelInfo["x1"]) / normalized_width
            x2 = float(labelInfo["x2"]) / normalized_width
            y1 = float(labelInfo["y1"]) / normalized_height
            y2 = float(labelInfo["y2"]) / normalized_height
            xmins.append(x1)
            xmaxs.append(x2)
            ymins.append(y1)
            ymaxs.append(y2)
            classes_text.append(str.encode(labelInfo["tags"][0]))
            classes.append(__allTages.index(labelInfo["tags"][0]) + 1)
        filename = str.encode(visitedFrameNames[frames_index])
        fileformat = str.encode(visitedFrameNames[frames_index].split('.')[1])
        
        if isTrain:
            __trainLabels.append(create_tf_example(\
                parseExampleObject(img_height, img_width\
                    , filename, fileformat\
                    , xmins, xmaxs, ymins, ymaxs, classes_text, classes) \
                , getImgEncode(crtFileName)))
        else:
            __evalLabels.append(create_tf_example(\
                parseExampleObject(img_height, img_width\
                    , filename, fileformat\
                    , xmins, xmaxs, ymins, ymaxs, classes_text, classes) \
                , getImgEncode(crtFileName)))

                
def prepareExampleList():
    allJsonFiles = getAllFileList(labelDir)
    for example in allJsonFiles:
        # get all tages
        conserveTagName(getJsonFile(example)['inputTags'].split(','))
        # write out the pbtxt
        writePbtxt(outputLabelFile)
        
        # get all file information
        getJsonContent = getJsonFile(example)
        sepTrainValidateData(getJsonContent, example, isTrain=True)
        
        if not math.isclose(1.0, TRAIN_VALIDARION_RATIO, rel_tol=1e-5):
            sepTrainValidateData(getJsonContent, example, isTrain=False)
            
def checkOutputPath():
    global outputFilePath
    if not os.path.isdir(outputFilePath):
        try:
            os.mkdir(outputFilePath)
            return 0
        except:
            return 1
    else:
        return 0

def main(_):
    prepareExampleList()
    
    # write training tfrecords
    with tf.python_io.TFRecordWriter(outputTrainTFRecordFile) as writer:
        for i in range(0, len(__trainLabels), 1):
            writer.write(__trainLabels[i].SerializeToString())
    
    if not math.isclose(1.0, TRAIN_VALIDARION_RATIO, rel_tol=1e-5):
        # write evaluating tfrecords
        with tf.python_io.TFRecordWriter(outputEvalTFRecordFile) as writer:
            for i in range(0, len(__evalLabels), 1):
                writer.write(__evalLabels[i].SerializeToString())
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '--labeldir',\
        type=str,\
        default=os.path.join('.','raw','origindata'),\
        help='labelling directory'\
    )
    parser.add_argument(\
        '--outputfilepath',\
        type=str,\
        default=os.path.join('.','data'),\
        help='output file for labelling data and tfrecords')
    
    #TRAIN_VALIDARION_RATIO = 0.8  # TRAIN:VALIDARION = 8:2
    parser.add_argument(\
        '--trainevalratio',\
        type=float,\
        default=0.8,\
        help='the ratio for train:evalution')
    FLAGS, unparsed = parser.parse_known_args()
    
    labelDir = FLAGS.labeldir
    outputFilePath = FLAGS.outputfilepath
    outputLabelFile = os.path.join(outputFilePath, 'image_label.pbtxt')
    outputTrainTFRecordFile = os.path.join(outputFilePath, 'train.tfrecords')
    outputEvalTFRecordFile = os.path.join(outputFilePath, 'eval.tfrecords')
    TRAIN_VALIDARION_RATIO = FLAGS.trainevalratio
    
    if checkOutputPath() == 0:
        tf.app.run()
    else:
        print("Error: Can not find or create the folder {}.".format(outputFilePath))














