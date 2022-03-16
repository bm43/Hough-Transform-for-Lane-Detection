######################################
### THIS SHOULD YIELD 7 .TXT FILES ##
######################################


import numpy as np
import os
import matplotlib.pyplot as plt
import math
import pcl
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor

import warnings
import subprocess

class PolynomialRegression_deg3(object):
    def __init__(self, degree=3, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs' :self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


class PolynomialRegression_deg1(object):
    def __init__(self, degree=1, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs' :self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

##############################
# simple threshold functions #
##############################

def FilterIntentisyThr(pc,thr=10):
    new_pc=[]
    for p in pc:
        #if thr2>=p[3]>=thr1:
        if p[3]>=thr:
        #if p[3] == thr1:
            new_pc.append(p)
    return np.array(new_pc)

def FilterROI(pc, yr1=-5, yr2=5):
    new_pc=[]
    yrange=[yr1,yr2]
    xrange=[-200,20]
    for p in pc:
        if yrange[0]<p[1]<yrange[1] and xrange[0]<p[0]<xrange[1]:
            new_pc.append(p)
    return np.array(new_pc)

def FilterROI7(pc): # arbitrary ROI for frame 7
    new_pc=[]
    for p in pc:
        if -p[0]-6<p[1]<-p[0]+5:
            new_pc.append(p)
    return np.array(new_pc)

def FilterLidarVal(pc,thr=48):
    new_pc=[]
    #lidar_values=[]
    for p in pc:
        #if p[4] % 2 == 1 and p[4] > 50:
        if p[4] >= thr: # first 16 layers, 48 ~ 63
        #if p[4] == 63:
            #lidar_values.append(p[4])
            new_pc.append(p)
    #print(set(lidar_values))
    return np.array(new_pc)
# frame 2, 3, 4, 7, 8, 9 and 10 look easiest

###################
# other functions #
###################

def separate_LR(pc,alones=True):
    pc = np.array(sorted(pc, key = lambda pc: pc[4], reverse = False))
    left_lane_b = np.array([])
    left_lane_f = np.array([])
    right_lane_b = np.array([])
    right_lane_f = np.array([])
    left_lane = np.array([])
    right_lane = np.array([])

    # if there is only one cluster, put them in alones
    front_alones = np.array([])
    back_alones = np.array([])


    for lidar_val in set(pc[:,4]): # 16 times max. circle getting bigger and bigger 48, 49, ... 63
        #print(set(pc[:,4]))
        points_with_same_lidar_val = np.array([ element for element in pc if element[4] == lidar_val ]) # this should look like a circle or arc
        thetas = np.array([np.arctan(point[1]/point[0]) for point in points_with_same_lidar_val])
        #radiuses = np.array(math.sqrt(point[0]**2+point[1]**2) for point in points_with_same_lidar_val])
        dim = points_with_same_lidar_val.shape
        new = np.zeros([dim[0],dim[1]+1])
        new[:,:-1] = points_with_same_lidar_val
        new[:,5] = thetas
        #new[:,6] = radiuses

        front = np.array([element for element in new if element[0] < 0])
        back = np.array([element for element in new if element[0] > 0])

        if len(front) > 0:
            if np.std(front[:,5]) > 0.1: # many clusters

                mu_f = np.mean(front[:,5])
                # using mean could be improved by using better cluster algorithms
                left_lane_f = np.append( left_lane_f, np.array([element for element in front if element[5] > mu_f]) ).reshape(-1,6)
                right_lane_f = np.append( right_lane_f, np.array([element for element in front if element[5] < mu_f]) ).reshape(-1,6)
            else: # one cluster, =small std
                front_alones = np.append(front_alones, front)
        else: # front is empty
            pass

        if len(back) > 0:
            if np.std(back[:,5]) > 0.1:
                mu_b = np.mean(back[:,5])
                left_lane_b = np.append( left_lane_b, np.array([element for element in back if element[5] < mu_b]) ).reshape(-1,6)
                right_lane_b = np.append( right_lane_b, np.array([element for element in back if element[5] > mu_b]) ).reshape(-1,6)
            else: # one cluster, =small std
                back_alones = np.append(back_alones, back)
        else: # back is empty
            pass

        right_lane = np.append(right_lane_b,right_lane_f).reshape(-1,6)
        left_lane = np.append(left_lane_b, left_lane_f).reshape(-1,6)
    if alones:
        # dealing with single clusters
        front_alones = front_alones.reshape(-1,6)
        back_alones = back_alones.reshape(-1,6)
        right_mean_y = np.mean(right_lane[:,1]) # mean y position of right lane
        left_mean_y = np.mean(left_lane[:,1]) # mean y position of left lane
        for p in back_alones:
            if abs(p[1]-right_mean_y) > abs(p[1]-left_mean_y): # dist to right is larger than dist to left
                left_lane = np.append(left_lane,p).reshape(-1,6)
            else:
                right_lane = np.append(right_lane,p).reshape(-1,6)
        for p in front_alones:
            if abs(p[1]-right_mean_y) > abs(p[1]-left_mean_y): # dist to right is larger than dist to left
                left_lane = np.append(left_lane,p).reshape(-1,6)
            else:
                right_lane = np.append(right_lane,p).reshape(-1,6)

    return left_lane, right_lane

def poly_fit_ransac(left_lane, right_lane, degree_l, degree_r, L_thr, R_thr, full_path_to_txt):

    xl = left_lane[:,0]
    yl = left_lane[:,1]
    xr = right_lane[:,0]
    yr = right_lane[:,1]

    #plt.scatter(xl,yl)
    # left:

    if degree_l == 1:
        ransac = RANSACRegressor(PolynomialRegression_deg1(),
                                 residual_threshold = L_thr*np.std(yl),
                                 random_state=0)
    else:
        ransac = RANSACRegressor(PolynomialRegression_deg3(),
                                 residual_threshold = L_thr*np.std(yl),
                                 random_state=0)
    ransac.fit(np.expand_dims(xl, axis=1), yl)
    #inlier_mask = ransac.inlier_mask_
    yl_hat = ransac.predict(np.expand_dims(xl, axis=1))
    #plt.plot(xl[inlier_mask], yl[inlier_mask], 'go', label='inliers (STD)')
    #plt.plot(xl, yl_hat, 'ro', label='estimated curve')
    c1, c2, c3, c4 = np.polyfit(xl,yl_hat,3)

    #plt.show()


    plt.scatter(xr,yr,label='original pc')
    #plt.legend()
    #plt.show()
    # right:
    if degree_r == 1:
        ransac = RANSACRegressor(PolynomialRegression_deg1(),
                                 residual_threshold = R_thr*np.std(yl),
                                 random_state=0)
    else:
        ransac = RANSACRegressor(PolynomialRegression_deg3(),
                                 residual_threshold = R_thr*np.std(yl),
                                 random_state=0,stop_n_inliers=40)
    ransac.fit(np.expand_dims(xr, axis=1), yr)

    inlier_mask = ransac.inlier_mask_
    yr_hat = ransac.predict(np.expand_dims(xr, axis=1))
    plt.plot(xr[inlier_mask], yr[inlier_mask], 'yo', label='inliers (STD)')
    plt.plot(xr, yr_hat, 'ro', label='estimated curve')
    c5, c6, c7, c8 = np.polyfit(xr,yr_hat,3)
    plt.legend()
    plt.show()

    '''
    f = open(full_path_to_txt,"w+")
    f.write(str(c1)+';'+str(c2)+';'+str(c3)+';'+str(c4)+'\n')
    f.write(str(c5)+';'+str(c6)+';'+str(c7)+';'+str(c8))
    '''

    return

def write_pc_to_txt(pc,full_path_to_txt,separator):

    f = open(full_path_to_txt,"w+")
    for p in pc:
        f.write(str(p[0])+separator+str(p[1])+separator+str(p[2])+'\n')
    return

def dat_to_txt_lines(full_path_to_dat,full_path_to_line):
    import re
    all_lines=[]
    try:
        with open (full_path_to_dat, "r") as myfile:
            data = myfile.read()
        data = data.split('\n')
    except FileNotFoundError:
        return
    f=open(full_path_to_line,"w+")
    for line in data:
        vals = re.split('[)(,]', line)
        print(vals)
        if len(vals) == 1:
            pass
        else:
            slope = float(vals[8])/float(vals[7])
            intercept = float(vals[3]) - slope * float(vals[2])
            all_lines.append([slope, intercept])
    all_lines = sorted(all_lines, key = lambda x: x[1], reverse=False)

    for i in range(2):
        f.write(str(0)+';'+str(0)+';'+str(all_lines[i][0])+';'+str(all_lines[i][1])+'\n')
    return

if __name__ == "__main__":


    warnings.filterwarnings("ignore")


    # folder with point cloud files:
    data_folder = "C:/Users/SamSung/Desktop/projects2/Seoul_Robotics_lane_detection/Seoul_Robotics_lane_detection/pointclouds/"
    files = [data_folder+path for path in sorted(os.listdir(data_folder))]

    # point cloud data in .txt files with comma as separators (only for hough transform
    pre_hough_folder = "C:/Users/SamSung/Desktop/projects2/Seoul_Robotics_lane_detection/for_submission/pc_in_txt_after_filter_comma/"

    # output .dat files, result of hough transform
    post_hough_folder = "C:/Users/SamSung/Desktop/projects2/Seoul_Robotics_lane_detection/for_submission/output_lines_after_filter/"

    # final coefficients in .txt files computed using output of hough transform
    final_output_folder = "C:/Users/SamSung/Desktop/projects2/Seoul_Robotics_lane_detection/for_submission/sample_output/"

    # folder containing hough3dlines.exe file
    hough_folder = "C:/Users/SamSung/Desktop/projects2/Seoul_Robotics_lane_detection/for_submission/hough3d-code/"


    '''
    # point cloud 1 = frame 0
    file_idx = 0
    pc = np.fromfile(files[file_idx], dtype=np.float32).reshape(-1, 5) # last frame
    filename = files[file_idx].split('/')[-1][:-4]
    pc = FilterROI(pc) # y between -3 and 3
    pc = FilterIntentisyThr(pc,10) # road markers
    pc = FilterLidarVal(pc) # vicinity

    # separates left and right, return left_lane and right_lane
    se = separate_LR(pc)

    # polynomial fitting with sklearn, RANSAC
    poly_fit_ransac(se[0],se[1],1,3,0.05,0.1,final_output_folder+filename+".txt") # fits poly, creates and writes to the txt in the folder



    # point cloud 2 = frame 1
    file_idx = 1
    pc = np.fromfile(files[file_idx], dtype=np.float32).reshape(-1, 5) # last frame
    filename = files[file_idx].split('/')[-1][:-4]
    pc = FilterROI(pc) # y between -3 and 3
    pc = FilterIntentisyThr(pc,10) # road markers
    pc = FilterLidarVal(pc) # vicinity
    # save pc as txt with commas:
    write_pc_to_txt(pc,pre_hough_folder+filename+".txt",',')

    # input txt point cloud path:
    input_pc_full_path = pre_hough_folder+filename+".txt"

    # where hough transform result will be written:
    output_dat_full_path = post_hough_folder+filename+".dat"

    # running hough3dlines.exe with parameters, input and output files:
    cmd_line = hough_folder+"hough3dlines.exe -nlines 3 -minvotes 15 "+input_pc_full_path+" -o "+output_dat_full_path

    subprocess.run(cmd_line) # dat file is created to output dat full path

    # hough transform result file transformed to .txt file to be used in data_visualize.py
    dat_to_txt_lines(output_dat_full_path,final_output_folder+filename+".txt")



    # point cloud 3 = frame 2
    file_idx = 2
    pc = np.fromfile(files[file_idx], dtype=np.float32).reshape(-1, 5) # last frame
    filename = files[file_idx].split('/')[-1][:-4]
    pc = FilterROI(pc)
    pc = FilterIntentisyThr(pc,10)
    pc = FilterLidarVal(pc) # vicinity
    write_pc_to_txt(pc,pre_hough_folder+filename+".txt",',')
    input_pc_full_path = pre_hough_folder+filename+".txt"
    output_dat_full_path = post_hough_folder+filename+".dat"
    cmd_line = hough_folder+"hough3dlines.exe -nlines 3 -minvotes 10 "+input_pc_full_path+" -o "+output_dat_full_path
    subprocess.run(cmd_line) # dat file is created to output dat full path
    dat_to_txt_lines(output_dat_full_path,final_output_folder+filename+".txt")



    # point cloud 5 = frame 4
    file_idx = 4
    pc = np.fromfile(files[file_idx], dtype=np.float32).reshape(-1, 5) # last frame
    filename = files[file_idx].split('/')[-1][:-4]
    pc = FilterROI(pc)
    pc = FilterIntentisyThr(pc) # road markers
    pc = FilterLidarVal(pc) # vicinity
    write_pc_to_txt(pc,pre_hough_folder+filename+".txt",',')
    input_pc_full_path = pre_hough_folder+filename+".txt"
    output_dat_full_path = post_hough_folder+filename+".dat"
    cmd_line = hough_folder+"hough3dlines.exe -nlines 2 -minvotes 15 "+input_pc_full_path+" -o "+output_dat_full_path
    subprocess.run(cmd_line) # dat file is created to output dat full path
    dat_to_txt_lines(output_dat_full_path,final_output_folder+filename+".txt")



    # point cloud 8 = frame 7
    file_idx = 7
    pc = np.fromfile(files[file_idx], dtype=np.float32).reshape(-1, 5) # last frame
    filename = files[file_idx].split('/')[-1][:-4]
    pc = FilterROI7(pc)
    pc = FilterIntentisyThr(pc,10) # road markers
    pc = FilterLidarVal(pc,45) # vicinity
    se = separate_LR(pc)
    poly_fit_ransac(se[0],se[1],1,3,0.1,0.1,final_output_folder+filename+".txt")



    # point cloud 10 = frame 9 완성
    file_idx = 9
    pc = np.fromfile(files[file_idx], dtype=np.float32).reshape(-1, 5)
    filename = files[file_idx].split('/')[-1][:-4]
    pc = FilterROI(pc)
    pc = FilterIntentisyThr(pc,10) # road markers
    pc = FilterLidarVal(pc) # vicinity
    se = separate_LR(pc)
    poly_fit_ransac(se[0], se[1], 3, 3, 0.1, 1, final_output_folder+filename+".txt")
    '''


    # point cloud 11 = frame 10 완성
    file_idx = 10
    pc = np.fromfile(files[file_idx], dtype=np.float32).reshape(-1, 5) # last frame
    filename = files[file_idx].split('/')[-1][:-4]
    pc = FilterROI(pc,-3,5)
    pc = FilterIntentisyThr(pc,15) # road markers
    pc = FilterLidarVal(pc) # vicinity
    se = separate_LR(pc)
    poly_fit_ransac(se[0],se[1],3,3, 1, 0.1, final_output_folder+filename+".txt")
    # frame 0, 1, 2, 4, 7, 9, 10
