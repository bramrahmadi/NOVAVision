from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unify
import numpy as np
import os
import sys
import argparse

def main(args):
    
    predictpaths = get_folders(args.predict_dir)
    gtpath1 = os.path.expanduser(args.gt_dir1)
    gtpath2 = os.path.expanduser(args.gt_dir2)

    gt_paths1, gt_names1 = unify.get_filenames(gtpath1)
    gt_paths2, gt_names2 = unify.get_filenames(gtpath2)
    

    prob = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']

    # for path in predictpaths:
    #     print(path)

    for path in predictpaths:
        
        if path[-1] == "1":
            gt_paths, gt_names = gt_paths2, gt_names2
        elif path[-1] == "2":
            gt_paths, gt_names = gt_paths1, gt_names1

        for p in prob:
            val = open(os.path.join(path, "met"+p+".txt"), 'w+')
            gtlabels = []
            prdlabels = []
            FN, FPd, TPd, FPs, TPs = 0,0,0,0,0
            sumIOU = []

            for seq in range(len(gt_paths)):
                filename = os.path.join(path, gt_names[seq]+p+".txt")
                print(filename)
                prd = open(filename, 'r')
                
                print(gt_paths[seq])
                gt = open(gt_paths[seq], "r")
                
                for line in gt:
                    
                    lineprd = prd.readline()
                    lineprd = lineprd.rstrip()
                    linegt = line.rstrip()

                    lineprd = lineprd.split(",")
                    linegt = linegt.split(",")

                    if lineprd[0] == linegt[0]:
                        bbgt = get_bbox(linegt)
                        bbprd = get_bbox(lineprd)

                        if bbprd == [0,0,0,0]:
                            FN+=1
                        else:
                            iou = get_IOU(bbgt,bbprd)
                            if iou > 0.2:
                                if linegt[5] != "0000":
                                    if (linegt[5] == lineprd[5]):
                                        TPs+=1
                                TPd+=1
                                
                            elif iou < 0.2:
                                if linegt[5] != "0000":
                                    if (linegt[5] == lineprd[5]):
                                        FPs+=1
                                FPd+=1
                            
                            sumIOU.append(iou)
                        
                        if linegt[5] != "0000":
                            gtlabels.append(int(linegt[5]))
                            prdlabels.append(int(lineprd[5]))
                
                prd.close()
                gt.close()

            avIOU = np.mean(sumIOU)
            precisionD = divide(TPd, (TPd+FPd))
            recallD = divide(TPd, (TPd+FN))
            precisionS = divide(TPs, (TPs+FPs))
            recallS = divide(TPs, (TPs+FN))
            acc = divide(TPs,(TPs+FPs+FN))

            print(avIOU)
            print(acc)

            val.write("{},{},{}\r\n".format(avIOU,precisionD,recallD))
            val.write("{},{},{},{},{}\r\n".format(TPs,FPs,FN,precisionS,recallS))
            val.write("{}\r\n".format(acc))

            val.close()


def divide(n, d):
    return n/d if d else 0

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

def get_folders(path):
    path_exp = os.path.expanduser(path)
    folders = []
    for root, dirs, files in os.walk(path_exp):
        for name in dirs:
            if os.path.isdir(os.path.join(root, name)):
                folders.append(os.path.join(root, name))
                
    return folders

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--predict_dir", type=str, default="~/Documents/TA/CODE/ChokePoint/prediction/")
    parser.add_argument("--gt_dir1", type=str, default="~/Documents/TA/CODE/ChokePoint/gt/G1/")
    parser.add_argument("--gt_dir2", type=str, default="~/Documents/TA/CODE/ChokePoint/gt/G2/")

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))