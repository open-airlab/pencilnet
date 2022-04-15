import json
import os
import cv2
import numpy as np
from shutil import copyfile


def display_target(M, source_img, ret=False):
    img = source_img.copy()
    img_height,img_width = img.shape[:2]
    grid_dim_x = img_width/5
    grid_dim_y = img_height/5

    bbox_results = []
    for i in range(5):
        for j in range(5):
            #cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
            #cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

            if M[i,j,0] > 0.5:
                cx, cy, rw, rh, distance, yaw_relative = M[i,j,1:]

                #print(M[i,j,0],i,j,cx,cy,rh)

                cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,255), 3)                

                h = int(rh*img_height)
                w = int(rw*img_width)

                cv2.rectangle(img, (int(cx_on_img-(w/2)), int(cy_on_img-(h/2))), 
                                    (int(cx_on_img+(w/2)), int(cy_on_img+(h/2))), (0,0,255), 3)

                # Unnormalize
                distance = distance*20.
                yaw_relative = (yaw_relative*np.pi)-np.pi                    

                yaw_relative = yaw_relative/np.pi*180.

                rel_text = "{:3.2f}".format(distance)+" meters"
                yaw_text = "{:3.2f}".format(yaw_relative)+" degree" 

                cv2.putText(img, rel_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                cv2.putText(img, yaw_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                bbox_results.append((cx_on_img,cy_on_img,abs(float(distance)),yaw_relative))
    #cv2.imwrite("display.png", img)
    if ret:
        return img, bbox_results


def display_target_woWH(M, source_img, output_shape, threshold, ret=False):
    img = source_img.copy()
    img_height,img_width = img.shape[:2]

    bbox_results = []
    nrow, ncol = output_shape[0], output_shape[1]
    grid_dim_x = img_width/ncol
    grid_dim_y = img_height/nrow


    for i in range(3):
        for j in range(nrow):
            #cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
            #cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

            if M[i,j,0] > threshold:
                cx, cy, distance, yaw_relative = M[i,j,1:]

                #print(M[i,j,0],i,j,cx,cy,rh)

                cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,1), 3)                

                #h = int(rh*img_height)
                #w = int(rw*img_width)

                #cv2.rectangle(img, (int(cx_on_img-(w/2)), int(cy_on_img-(h/2))), 
                #                    (int(cx_on_img+(w/2)), int(cy_on_img+(h/2))), (0,0,255), 3)

                # Unnormalize
                #distance = distance*20.
                #yaw_relative = yaw_relative*np.pi
                #yaw_relative = (yaw_relative+np.pi)/(np.pi)                    

                #yaw_relative = yaw_relative/np.pi*180.

                #rel_text = "{:3.2f}".format(distance)+" meters"
                #yaw_text = "{:3.2f}".format(yaw_relative)+" degree" 

                #cv2.putText(img, rel_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                #cv2.putText(img, yaw_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                bbox_results.append((cx_on_img,cy_on_img,abs(float(distance)),yaw_relative))
                
    #cv2.imwrite("display.png", img)
    if ret:
        return img, bbox_results
