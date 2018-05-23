# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 23:43:33 2018
@author: acer4755g
@reference: 
"""

import tensorflow as tf
import os
import Vott2TFRecordAndTFLabel_multiple 
import matplotlib.pyplot as plt

outputFilePath = Vott2TFRecordAndTFLabel_multiple.outputFilePath
outputLabelFile = Vott2TFRecordAndTFLabel_multiple.outputLabelFile
TRAIN_VALIDARION_RATIO = Vott2TFRecordAndTFLabel_multiple.TRAIN_VALIDARION_RATIO

#
# desc : read and decode tfrecord
# param@
# |- filename_queue: input a filename queue
#
def read_and_decode(filename_queue):
    # create a tfrecord object
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # decode the example
    features = tf.parse_single_example(serialized_example,\
                features={'image/encoded':tf.FixedLenFeature([], tf.string)\
                          , 'image/object/class/label':tf.VarLenFeature(tf.int64)\
                          , 'image/object/bbox/xmax':tf.VarLenFeature(tf.float32)\
                          , 'image/height':tf.FixedLenFeature([], tf.int64) \
                          , 'image/width':tf.FixedLenFeature([], tf.int64)})
    
    label = tf.cast(features['image/object/class/label'], tf.int64)
    label = tf.sparse_tensor_to_dense(label)
    xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
    xmax = tf.sparse_tensor_to_dense(xmax)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    
    # it must decode byteslist from string type to uint8 type
    image = tf.image.decode_jpeg(features['image/encoded'])
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image, height, width, label, xmax
    
#
# desc : input training or validation file (could be shuffle) and return a tuple
#
def inputs(data_set_name, num_epochs=None, outputImage=False):
    with tf.name_scope('input'):
        # return a QueueRunner object and FIFOQueue object inside in
        filename_queue = tf.train.string_input_producer([data_set_name], num_epochs=num_epochs)
    
    image, height, width, label, xmax = read_and_decode(filename_queue)
    
    if outputImage:
        # output all images
        with tf.Session() as sess:
            
            # because one epoch variable is built inside string_input_produer (image_raw)
            # and the variable is belonging to tf.GraphKeys.LOCAL_VARIABLES
            # tf.local_variables_initializer() is necessary
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            coord=tf.train.Coordinator()
            threads= tf.train.start_queue_runners(coord=coord)
            
            for i in range(0, 3, 1):
                single, heg, wdt, lbl, xmn = sess.run([image, height, width, label, xmax])
    
                # show the image
                plt.imshow(single)
                plt.show()
                
                # print the label
                print("Image height:{}, width:{}, label:{}, xmax:{}.".format(heg, wdt, lbl, xmn))
                
            coord.request_stop()
            coord.join(threads) 
    
if __name__ == '__main__':    
    inputs(\
        os.path.join(outputFilePath,'train.tfrecords')\
        , outputImage=True)   
    















