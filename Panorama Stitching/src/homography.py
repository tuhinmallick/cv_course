import numpy as np
import cv2

## Compute a homography matrix from 4 point matches
def computeHomography(points1, points2):
    '''
    Solution with OpenCV calls not allowed
    '''
    assert(len(points1) == 4)
    assert(len(points2) == 4)

    ## TODO 3
    ## Construct the 8x9 matrix A.
    ## Use the formula from the exercise sheet.
    ## Note that every match contributes to exactly two rows of the matrix.



    U, s, V = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(V)

    H = np.zeros((3, 3))
    ## TODO 3
    ## - Extract the homogeneous solution of Ah=0 as the rightmost column vector of V.
    ## - Store the result in H.
    ## - Normalize H



    return H

def testHomography():
    points1 = [(1, 1), (3, 7), (2, -5), (10, 11)]
    points2 = [(25, 156), (51, -83), (-144, 5), (345, 15)]

    H = computeHomography(points1, points2)

    print ("Testing Homography...")
    print ("Your result:" + str(H))

    Href = np.array([[-151.2372466105457,   36.67990057507507,   130.7447340624461],
                 [-27.31264543681857,   10.22762978292494,   118.0943169422209],
                 [-0.04233528054472634, -0.3101691983762523, 1]])

    print ("Reference: " + str(Href))

    error = Href - H
    e   = np.linalg.norm(error)
    print ("Error: " + str(e))

    if (e < 1e-10):
        print ("Test: SUCCESS!")
    else:
        print ("Test: FAIL!")
    print ("============================")
