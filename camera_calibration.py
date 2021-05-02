import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image


# get H
def homography(objpoints, imgpoints):
    H = np.zeros((len(objpoints), 9))

    imgpoints = np.squeeze(imgpoints)   # (img_num, 49, 1, 2) -> (img_num, 49, 2)
    img_num = len(objpoints)
    for img_idx in range(img_num):    # for every image, each has 49 points
        pt_3d = objpoints[img_idx]
        pt_2d = imgpoints[img_idx]
        P = np.empty([0, 9])

        for pt_idx in range(len(pt_3d)):
            x = pt_3d[pt_idx, 0]
            y = pt_3d[pt_idx, 1]
            u = pt_2d[pt_idx, 0]
            v = pt_2d[pt_idx, 1]

            arr1 = np.array([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            arr2 = np.array([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
            P = np.concatenate((P, arr1[np.newaxis,:], arr2[np.newaxis,:],), axis=0)
        
        U, S, Vt = np.linalg.svd(P)

        H[img_idx] = Vt[-1] # Vt[np.argmin(S)]
        H[img_idx] /= H[img_idx][-1]    # normalize 
        
        # print('my homo: ', H[img_idx].reshape(3, 3))
        # pt_3 = pt_3d[:,:2]
        # pt_2 = pt_2d
        # h, status = cv2.findHomography(pt_3, pt_2)
        # print('built-in homo',h)

    H = np.reshape(H, (img_num, 3, 3))
    print('\nH:\n', H)
    return H
    

def get_V(h, i, j):
    return np.array([ h[0,i] * h[0,j],
                    h[0,i] * h[1,j] + h[1,i] * h[0,j],
                    h[1,i] * h[1,j],
                    h[2,i] * h[0,j] + h[0,i] * h[2,j],
                    h[2,i] * h[1,j] + h[1,i] * h[2,j],
                    h[2,i] * h[2,j]])


# get B, K
def intrinsic(H):

    # V = np.zeros([2 * len(H), 6])
    # idx = 0
    # for _h in H:
    #     V[idx] = get_V(_h, 0, 1)
    #     V[idx + 1] = get_V(_h, 0, 0) - get_V(_h, 1, 1)
    #     idx += 2

    # U, S, Vt = np.linalg.svd(V)
    # b = Vt[-1]  # Vt[np.argmin(S)]
    # B = np.array([[b[0], b[1], b[3]],
    #              [b[1], b[2], b[4]],
    #              [b[3], b[4], b[5]]])
    
    # lambda_B = b[5] - ( b[3]*b[3] + pow((b[0]*b[4] - b[1]*b[3]), 2) / (b[0]*b[2]-b[1]*b[1])) / b[0]
    # B = B / lambda_B    # normalize

    # K = np.linalg.inv(np.linalg.cholesky(B).T)
    # K_inv = np.linalg.inv(K)
    # print("\nK:\n", K)

    # return K

    K_arr = []
    for l in range(2, len(H)+1):
        print(l)
        H_sub = H[:l]
        V = np.zeros([2 * len(H_sub), 6])
        idx = 0
        for _h in H_sub:
            V[idx] = get_V(_h, 0, 1)
            V[idx + 1] = get_V(_h, 0, 0) - get_V(_h, 1, 1)
            idx += 2

        U, S, Vt = np.linalg.svd(V)
        b = Vt[-1]  # Vt[np.argmin(S)]
        B = np.array([[b[0], b[1], b[3]],
                     [b[1], b[2], b[4]],
                     [b[3], b[4], b[5]]])
        
        lambda_B = b[5] - ( b[3]*b[3] + pow((b[0]*b[4] - b[1]*b[3]), 2) / (b[0]*b[2]-b[1]*b[1])) / b[0]
        B = B / lambda_B    # normalize

        K = np.linalg.inv(np.linalg.cholesky(B).T)
        K_inv = np.linalg.inv(K)
        print("\nK:\n", K)
        K_arr.append(K)

    return K_arr


# get R, t
def extrisic(H, K_arr):

    extrinsics = np.empty([0, 3, 4])
    # K_inv = np.linalg.inv(K)
    for img_idx in range(len(H)):
        K_inv = np.linalg.inv(K_arr[img_idx-1]) if img_idx > 0 else np.linalg.inv(K_arr[0])
        #  = np.linalg.inv(K)
        h1 = H[img_idx][:, 0]
        h2 = H[img_idx][:, 1]
        h3 = H[img_idx][:, 2]
        
        lambda1 = 1 / np.linalg.norm(np.dot(K_inv, h1))
        lambda2 = 1 / np.linalg.norm(np.dot(K_inv, h2))
        lambda3 = (lambda1 + lambda2) / 2
        
        r1 = lambda1 * np.dot(K_inv, h1)
        r2 = lambda2 * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)
        t = lambda3 * np.dot(K_inv, h3)

        Rt = np.array([r1.T, r2.T, r3.T, t.T]).T
        extrinsics = np.concatenate((extrinsics, Rt[np.newaxis,:]), axis=0)

    print('\nextrinsic:\n', extrinsics)
    return extrinsics


def distance(mat1, mat2):
    return np.linalg.norm(mat1-mat2)


### main ###
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('images/*.JPG')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)      # (img_num, 49, 3)
        imgpoints.append(corners)   # (img_num, 49, 1, 2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)



#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])

H = homography(objpoints, imgpoints)
K_arr = intrinsic(H)
extrinsics = extrisic(H, K_arr) # Rt

#print(K)



### check if correct ###
check_idx = 1
print('my final: ', extrinsics[check_idx])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsicsss = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
RRR, _ = cv2.Rodrigues(extrinsicsss[check_idx,0:3])
TTT = extrinsicsss[check_idx,3:6]
print('original final: ', RRR, TTT)

dist_arr = []
for i in range(len(K_arr)):
    dist_arr.append(distance(K_arr[i], mtx))
idx = np.arange(5, len(K_arr)+2)

fig = plt.figure()
plt.plot(idx, dist_arr[3:])
plt.xticks(idx)
plt.ylim(0, 500)
plt.xlabel('images')
plt.ylabel('distance')
plt.title('Experiment')
plt.savefig('experiment.png')





# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K_arr[-1]
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
# plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
