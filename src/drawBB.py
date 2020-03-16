from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import cv2 as cv
import os
import numpy as np

def main():

    flags = ["H1","M1","L1"]

    predict_dir = "~/Documents/TA/CODE/ChokePoint/prediction"

    gt_dir = "~/Documents/TA/CODE/ChokePoint/gt/G2"

    save_dir = "~/Documents/TA/CODE/ChokePoint/sample"

    video_dir = "~/Documents/TA/CODE/ChokePoint/video"

    seq = "P1L_S3_C3"

    prob = ["0.0"]

    gt_path = os.path.expanduser(gt_dir)
    prd_path = os.path.expanduser(predict_dir)
    save_path = os.path.expanduser(save_dir)
    video_path = os.path.join(os.path.expanduser(video_dir),seq)

    for p in prob:
        maxFile = ""
        maxIOU = 0

        for f in flags:

            gtfile = open(os.path.join(gt_path,seq+".txt"), 'r')
            prdfile = open(os.path.join(prd_path,"{}/{}{}.txt".format(f,seq,p)), 'r')

            
            maxBBGT = []
            maxBBPRD = []
            
            if not maxFile:
                for line in gtfile:
                    prdline = prdfile.readline()
                    prdline = prdline.rstrip().split(',')
                    gtline = line.rstrip().split(',')

                    if prdline[0] == gtline[0]:

                        bbgt = get_bbox(gtline)
                        bbprd = get_bbox(prdline)

                        if bbprd != [0,0,0,0]:
                            
                            iou = get_IOU(bbgt,bbprd)
                            print(iou)
                            if iou > maxIOU:
                                
                                maxIOU = iou
                                maxBBGT = bbgt
                                maxBBPRD = bbprd
                                maxFile = gtline[0]
            else:
                for line in gtfile:
                    prdline = prdfile.readline()
                    prdline = prdline.rstrip().split(',')
                    gtline = line.rstrip().split(',')

                    if prdline[0] == maxFile:
                        bbgt = get_bbox(gtline)
                        bbprd = get_bbox(prdline)
                        maxIOU = get_IOU(bbgt, bbprd)
                        maxBBGT = bbgt
                        maxBBPRD = bbprd

            
            print(maxFile)
            img = cv.imread(os.path.join(video_path,maxFile+".jpg"))

            print(len(maxBBGT))

            cv.rectangle(img, (maxBBGT[1], maxBBGT[0]), (maxBBGT[2]+maxBBGT[1], maxBBGT[3]+maxBBGT[0]), (0, 255, 0), 2)
            cv.rectangle(img, (maxBBPRD[1], maxBBPRD[0]), (maxBBPRD[2]+maxBBPRD[1], maxBBPRD[3]+maxBBPRD[0]), (0, 0, 255), 2)
            cv.putText(img, str(round(maxIOU,2)), (maxBBPRD[1], maxBBPRD[0]-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)

            filename = os.path.join(save_path,"{}.jpg".format(f))

            face = img[maxBBGT[0]-70:maxBBGT[3]+maxBBGT[0]+70, maxBBGT[1]-70:maxBBGT[2]+maxBBGT[1]+70,:]

            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            
            cv.imwrite(filename, face)
            gtfile.close()
            prdfile.close()


def get_IOU(bb1, bb2):

    xA = max(bb1[1],bb2[1])
    yA = max(bb1[0],bb2[0])
    xB = min(bb1[2]+bb1[1],bb2[2]+bb2[1])
    yB = min(bb1[3]+bb1[0],bb2[3]+bb2[0])

    interArea = max(0, xB-xA+1) * max(0, yB-yA+1)

    bb1Area = (bb1[3]+1) * (bb2[2]+1)
    bb2Area = (bb2[3]+1) * (bb2[2]+1)

    iou = interArea / float(bb1Area+bb2Area - interArea)

    return iou

def get_bbox(lines):

    bb = []
    bb.append(int(lines[1]))
    bb.append(int(lines[2]))
    bb.append(int(lines[3]))
    bb.append(int(lines[4]))

    return bb

if __name__ == "__main__":
    main()