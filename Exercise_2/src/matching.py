import pdb
import cv2
from utils import createMatchImage

def matchknn2(descriptors1, descriptors2):

    ## Initialize an empty list of matches. (HINT: N x 2)
    knnmatches = []

    ## TODO 2.1
    ## Find the two nearest neighbors for every descriptor in image 1.

    ## For a given descriptor i in image 1:
    ## Store the best match (smallest distance) in knnmatches[i][0]
    ## Store the second best match in knnmatches[i][1]

    ## Hint: The hamming distance between two descriptors can be computed with
    ## double distance = norm(descriptors1.row(i), descriptors2.row(j),NORM_HAMMING);


    return knnmatches

def ratioTest(knnmatches, ratio_threshold):
    matches = []

    ## TODO 2.2
    ## Compute the ratio between the nearest and second nearest neighbor.
    ## Add the nearest neighbor to the output matches if the ratio is smaller than ratio_threshold.
    #if SOLUTION >= 2


    return matches

def computeMatches(img1, img2):
    knnmatches = matchknn2(img1['descriptors'], img2['descriptors'])
    matches = ratioTest(knnmatches, 0.7)
    print ("(" + str(img1['id']) + "," + str(img2['id']) + ") found " + str(len(matches)) + " matches.")
    return matches
