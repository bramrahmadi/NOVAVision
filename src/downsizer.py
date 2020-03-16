from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL
import numpy as np
import os
import sys
import cv2 as cv
import facenet
import argparse
from scipy import misc

from PIL import Image

def main(args):
    input_dir = os.path.expanduser(args.input_directory)
    output_dir = os.path.expanduser(args.output_directory)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset = facenet.get_dataset(input_dir)

    for cls in dataset:
        out_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        for image_path in cls.image_paths:
            print(image_path)
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            filename = cls.name+'/'+filename
            print(filename)
            down(image_path, args, filename)

def down(images, args, output):
    W = 500
    img = misc.imread(images)
    width, height, depth = img.shape
    print('%d %d %d' % (height, width, depth))
    imgScale = W/width
    newX, newY = img.shape[0]*imgScale, img.shape[1]*imgScale
    newimg = misc.imresize(img, (int(newX), int(newY)), interp='bilinear')
    # wpercent = basewidth / float(img.shape[1])
    # hsize = int(float(img.shape[0] * float(wpercent)))
    # img = cv.resize(img, (hsize, basewidth), interpolation=cv.INTER_LINEAR)
    filen = os.path.expanduser(os.path.join(args.output_directory, output+'.png'))
    misc.imsave(filen,newimg)
    # cv.imwrite(filen,newimg)

def parser_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_directory', type=str, default='~/Pictures/Photo/')
	parser.add_argument('--output_directory', type=str, default='~/Pictures/Photo_resized/')

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parser_arguments(sys.argv[1:]))