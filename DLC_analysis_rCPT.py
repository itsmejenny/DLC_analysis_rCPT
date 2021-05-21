 # -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:39:37 2021

@author: hbn698
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob



#%%

path = r'Q:/Common/Lab member folder/Jenny/DLC_h5files/'
DLCscorer='DLC_resnet50_GiDREADD_in_LCFeb19shuffle1_800000'

os.chdir(path)
h5_files = glob.glob('*.h5')
filepath = 'Q:/Common/Lab member folder/Jenny/DLC_Analysis/DLC_video_organization_matlab.csv'

Coordinates = pd.read_csv(filepath, sep=';')
Coordinates = Coordinates.dropna(1, 'all')

#%%
cols = ['percentage_screen', 'percentage_reward', 'time_in_ROI', 'total_frames', 'percentage_time_in_ROI', 'angle_and_ROI', 'percentage_angle_and_ROI']

Results = pd.DataFrame(columns=cols, index=range(len(h5_files)))
Coordinates_Results = Coordinates.join(Results)


for i in range(len(h5_files)):
    #load in the data, define variables
    Data = pd.read_hdf(path + h5_files[i])
    print(h5_files[i])
    filename_str = h5_files[i].split('DLC')[0]
    print(filename_str)
    Nose_x = Data[DLCscorer]['Nose']['x']
    Nose_y = Data[DLCscorer]['Nose']['y']
    L_F_x = Data[DLCscorer]['B_F_L']['x']
    L_F_y = Data[DLCscorer]['B_F_L']['y']
    R_F_x = Data[DLCscorer]['B_F_R']['x']
    R_F_y = Data[DLCscorer]['B_F_R']['y']
    L_B_x = Data[DLCscorer]['B_B_L']['x']
    L_B_y = Data[DLCscorer]['B_B_L']['y']
    R_B_x = Data[DLCscorer]['B_B_R']['x']
    R_B_y = Data[DLCscorer]['B_B_R']['y']
    shoulders_x =  np.nanmean([L_F_x ,R_F_x ],axis=0)
    shoulders_y =  np.nanmean([L_B_y ,R_B_y],axis=0)
    bodycenter_x = np.nanmean([L_F_x, R_B_x], axis=0)
    bodycenter_y = np.nanmean([L_F_y, R_B_y], axis=0)
    
    #create variable for time in frames
    frames = range(int(len(Data)))
    
    #calculating the direction of the head
    normX = Nose_x-shoulders_x
    normY = Nose_y-shoulders_y
    head_dir = (np.arctan2(normY, normX))
    head_dir_degree = (head_dir*180)/np.pi
    
    # plotting the head direction angles in radian and degree
    fig, axs = plt.subplots(2)
    axs[0].plot(frames, head_dir)
    axs[1].plot(frames, head_dir_degree)
    fig.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------    
    # collecting frames in which the specified criteria are hit (small angle = +- 20 degrees, reward = +-180 degrees +- 20)
    small_angle = []
    reward = []
    
    r = range(-20,21)
    
    for frame in frames:
        if int(head_dir_degree[frame]) in r:
            small_angle.append(head_dir_degree[frame])
        elif -180 < int(head_dir_degree[frame]) < -160:
            reward.append(head_dir_degree[frame])
        elif 160 < int(head_dir_degree[frame]) < 180:
            reward.append(head_dir_degree[frame])
    
    #print the results
    percentage_screen = len(small_angle)/len(frames)*100
    rounded_percentage = round(percentage_screen, 2)
    
    percentage_reward = len(reward)/len(frames)*100
    rounded_percentage_reward = round(percentage_reward, 2)
#-----------------------------------------------------------------------------------------------------------------------------------------------
    print('Filename: ' + h5_files[i])
    print('The estimated percentage of time looking at the screen is: ' + str(rounded_percentage) + ' %.')
    print('The estimated percentage of time looking towards the reward tray is: ' + str(rounded_percentage_reward) + ' %.')
    
   

    time_in_ROI = []
    row = Coordinates.loc[Coordinates['filename']==filename_str]
    row = row.reset_index(drop = True)
    bottom_left_x = row['bottom_left_x'].loc[0]
    bottom_left_y = row['top_right_y'].loc[0]
    top_right_x = row['top_right_x'].loc[0]
    top_right_y = row['bottom_left_y'].loc[0]
    bottom_right_x = row['bottom_right_x'].loc[0]
    bottom_right_y = row['bottom_right_y'].loc[0]
    top_left_x = row['top_left_x'].loc[0]
    top_left_y = row['top_left_y'].loc[0]
    
    # see if bodycenter is in ROI in front of screen
    def solve(bl, tr, p) :
        return(p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1])
           
    for frame in frames:
        bottom_left = (bottom_left_x, bottom_left_y)
        top_right = (top_right_x, top_right_y)
        point = (int(bodycenter_x[frame]),int(bodycenter_y[frame]))
        result = solve(bottom_left, top_right, point)
        if result:
            time_in_ROI.append(result)
    
    percentage_time_in_ROI = len(time_in_ROI)/len(frames)*100
   
    print('Coordinates for the ROI: ' + str(bottom_left) + str(top_right))
    print('Time spent in ROI in front of the screen in frames:')
    print(len(time_in_ROI))
    print('Percentage of frames where mouse is in front of the screen:')
    print(percentage_time_in_ROI)


    angle_and_ROI = []
        
    for frame in frames:
        point = (int(bodycenter_x[frame]),int(bodycenter_y[frame]))
        if int(head_dir_degree[frame]) in r and solve(bottom_left, top_right, point):
            angle_and_ROI.append(head_dir_degree[frame])
    
    print('Percentage of frames where mouse is in the ROI in front of the screen and "looking" at the screen:')
    percentage_angle_and_ROI = len(angle_and_ROI)/len(frames)*100
    print(percentage_angle_and_ROI)
    
    bpt = 'Nose'
    pcutoff = 0.9
    Index = Data[DLCscorer][bpt]['likelihood'].values < pcutoff
    xnose=Data[DLCscorer][bpt]['x'].values
    xnose[Index]=np.nan
    ynose=Data[DLCscorer][bpt]['y'].values
    ynose[Index]=np.nan
    
    figure, axis = plt.subplots(1)
    plt.plot(xnose, ynose,'.-', alpha=0.3, zorder=1, color= 'gray')
    import matplotlib.patches as patches
    rect = patches.Rectangle(bottom_left, top_right_x - bottom_left_x, top_right_y - bottom_left_y ,linewidth=1,edgecolor='red',facecolor='none', zorder = 2)
    axis.add_patch(rect)
    plt.xlim([-50, 800])
    plt.ylim([-50,600])
    plt.show()
    
    Coordinates_Results.loc[Coordinates_Results.filename==filename_str, 'percentage_screen'] =  percentage_screen
    Coordinates_Results.loc[Coordinates_Results.filename==filename_str, 'percentage_reward'] = percentage_reward
    Coordinates_Results.loc[Coordinates_Results.filename==filename_str, 'time_in_ROI'] = len(time_in_ROI)
    Coordinates_Results.loc[Coordinates_Results.filename==filename_str, 'percentage_time_in_ROI'] = percentage_time_in_ROI
    Coordinates_Results.loc[Coordinates_Results.filename==filename_str, 'total_frames'] = len(frames)
    Coordinates_Results.loc[Coordinates_Results.filename==filename_str, 'angle_and_ROI'] = len(angle_and_ROI)
    Coordinates_Results.loc[Coordinates_Results.filename==filename_str, 'percentage_angle_and_ROI'] = percentage_angle_and_ROI

    
    print('-----------------------------------------------------')

Coordinates_Results.to_csv(r'Q:/Common/Lab member folder/Jenny/DLC_analysis_results.csv', sep = ',', na_rep='nan', index = False, header = True)
Coordinates_Results.to_excel(r'Q:/Common/Lab member folder/Jenny/DLC_analysis_results.xlsx', na_rep='nan', index = False, header = True)
