from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import xml.etree.ElementTree as ET

def main(args):

    bbpath = os.path.expanduser(args.bb_dir)
    gtpath = os.path.expanduser(args.gt_dir)
    bbfiles, bbnames = get_filenames(bbpath)
    print("OK")
    for i in range(len(bbfiles)):
        bbfile = open(bbfiles[i], "r")
        print(bbfiles[i]+", "+bbnames[i])
        tree = ET.parse(os.path.join(gtpath,bbnames[i]+".xml"))
        root = tree.getroot()
        newfile = open(os.path.join(gtpath,bbnames[i]+".txt"), "w+")
        for line in bbfile:
            fields = line.split(",")
            frame = fields[0]
            newline = line.rstrip()
            # ids = tree.find("//frame[@number='%s' or .//*[@number='%s']]/person/@id" % (frame,frame))
            # ids = tree.find("//frame[@number='%s']/person" % frame)
            # idx = ids.get['id']
            idnew = get_ID(root, frame)
            newfile.write("%s,%s\r\n" % (newline, idnew))
            print("%s,%s\r\n" % (newline, idnew))
        bbfile.close()
        newfile.close()
        root.clear()

def get_filenames(path):
    filedir = []
    names = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".txt"):
                filedir.append(os.path.join(path,f))
                names.append(f[:-4])
    return filedir, names

def get_ID(tree, framenum):
    for x in tree.findall('frame'):
        # print(x.get('number'))
        if x.get('number') == framenum:
            idx = x.find('person')
            if idx is None:
                # print("NO")
                ids = "0000"
            else:
                # print("OK")
                ids = idx.get('id')

    return ids

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--bb_dir', type=str, default="~/Documents/TA/CODE/ChokePoint/groundtruth/G1/")
    parser.add_argument('--gt_dir', type=str, default="~/Documents/TA/CODE/ChokePoint/groundtruth/")

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))