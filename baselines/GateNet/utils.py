import json
import math

#PATH_TO_DATASET = "/gate_pose_dataset/"

def read_annot(path_to_annot_file):
    with open(path_to_annot_file, 'r') as json_file:
        data = json.load(json_file)
        print(data.keys())
        print(data['classes'])
        for i in range(10):
            print(data['annotations'][i])
      

#read_annot(PATH_TO_DATASET+'training/'+'lights_hand_flight/annotations.json')        


import cv2
import os
import json
import shutil
import numpy as np
from shutil import copyfile


def recursive(folder):
    paths = []
    
    dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for d in dirs:
        path_dir = os.path.join(folder, d)
        if os.path.isfile(os.path.join(path_dir, "annotations.json")):
            paths.append(path_dir)

        else:
            paths += recursive(path_dir)

    return paths


def merge_hybrid_dataset(destination, source):

    if os.path.isdir(destination):
        print("[*] Destination folder: %s" % destination)
        if not os.path.isdir(os.path.join(destination, 'images')):
            os.mkdir(os.path.join(destination, 'images'))
    else:
        raise RuntimeError("Destination folder does not exist: %s" %destination)

    if os.path.isdir(source):
        print("[*] Source folder: %s" % source)
    else:
        raise RuntimeError("Source folder does not exist: %s" %source)

    cnt = 0

    data_folders = recursive(source)
    print("[*] Data folders: %s" % str(data_folders))


    annot_file = open(os.path.join(destination, "annotations.json"), "w+")
    content = {'annotations':[]}
    for data_folder in data_folders:

        print(os.path.join(data_folder, "annotations.json"))
        with open(os.path.join(data_folder, "annotations.json"), 'r') as json_file:
            data = json.load(json_file)

            for annot in data["annotations"]:

                img_name = annot["image"]
                img_path = os.path.join(data_folder, "images", img_name)
                print(img_path)

                annot['image']='{0:06d}'.format(cnt)+'.png'
                shutil.copyfile(img_path, os.path.join(destination, "images", annot['image']))

                content["annotations"].append(annot)
                cnt += 1
                #print(annot)            
    
    json.dump(content, annot_file)
    annot_file.close()

"""
destination = "huy-9Jan-including-partially-observed-gates-merged"
source = "DATASETS/huy_9Jan_including_partially_observed_gates"    

merge_hybrid_dataset(destination, source)
"""

"""
destination = "huy-9Jan-and-11Feb-merged-raw-orientation"
source = "huy-9Jan-and-11Feb-raw-orientation"    

merge_hybrid_dataset(destination, source)
"""
"""
destination = "DATASETS/17June-shifted-merged"
source = "DATASETS/17June-shifted"    

merge_hybrid_dataset(destination, source)
"""
"""
destination = "DATASETS/merged-combined-data"
source = "DATASETS/combined-data"    

merge_hybrid_dataset(destination, source)
"""

"""
destination = "DATASETS/merged-combined-data-cleaned"    
source = "/media/my_folder/DATASET_25June/"
merge_hybrid_dataset(destination, source)
"""

"""
destination = "DATASETS/merged-6July"    
source = "DATASETS/6July/"
merge_hybrid_dataset(destination, source)
"""

"""
destination = "DATASETS/merged-data-8July-including-intel"    
source = "DATASETS/data-8July/"
merge_hybrid_dataset(destination, source)
"""

"""
destination = "DATASETS/merged-data-8July-large-annot/"    
source = "DATASETS/data-8July-large-annot/"
merge_hybrid_dataset(destination, source)
"""

"""
destination = "DATASETS/merged-data-8July-large-annot-cond/"    
source = "DATASETS/data-8July-large-annot-cond/"
merge_hybrid_dataset(destination, source)
"""

"""
source = "DATASETS/"    
destination = "merged-data-8July-large-annot-cond-and-dark-8Sep/"
merge_hybrid_dataset(destination, source)
"""

"""
source = "DATASETS/night_data_handheld"    
destination = "DATASETS/night-dataset-3"
merge_hybrid_dataset(destination, source)
"""

"""
source = "DATASETS/au_dr_combine/"    
destination = "DATASETS/au_dr"
merge_hybrid_dataset(destination, source)
"""




class HUY2DATASET:
    """ Create labels for training using back projection. 
        Use vicon data for this.

        This class is a converter to 3D coordinates of gates to 2D pixel coordinates.
    """

    def __init__(self):
        # Create the projection matrix
        self.projection_matrix = np.array([[628.559937, 0.000000, 322.631874, 0.000000, 0.000000, 641.147705, 260.018180, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]]).T 
        self.projection_matrix = np.reshape(self.projection_matrix, (4,3), order='F').T

        # Estimated offsed between camera and UAV's COM
        self.camera_position_offset = np.array([0.05, -0.05, 0.02]) # in UAV body frame
        self.camera_rotation_offset = np.array([1.5, -1.9, -2.3]) / 180. * np.pi

    def run(self, huy_folder_name, dest_folder):

        print("[*] Huy folder: %s" % huy_folder_name)

        # Read label.txt file to a dictionary.   
        dataset = {} 
        with open(os.path.join(huy_folder_name, 'label.txt'), 'r') as label_file:
            label_names = label_file.readline().split(' ')
            label_names[-1] = label_names[-1][:-1]
            for name in label_names:
                dataset[name]=[]

            print("[*] Headers: %s" % str(label_names))
            for line in label_file.readlines():
                values = (line.split(' '))   
                values[-1] = values[-1][:-1]
                values = [values[0]] + [float(value) for value in values[1:]]
                for i in range(len(values)):
                    dataset[list(dataset.keys())[i]].append(values[i])
                
            
        num_of_samples = len(dataset[list(dataset.keys())[0]])
        print("[*] Num of samples found: %d" % num_of_samples)


        output = {"annotations":[]}
        for i in range(num_of_samples):

            # Gate postion in Vicon frame
            gate_position_3D = np.array([[dataset['gate_x'][i], dataset['gate_y'][i], dataset['gate_z'][i]]])
            gate_orientation = np.array([[dataset['gate_r'][i], dataset['gate_p'][i], dataset['gate_yaw'][i]]])
            

            
            gate_rotation = np.array([[np.cos(gate_orientation[0,2]), -np.sin(gate_orientation[0,2]), 0],
                                    [np.sin(gate_orientation[0,2]), np.cos(gate_orientation[0,2]), 0],
                                    [0, 0, 1]])

            # Gate corners postion in Vicon frame
            corners_position_3D = np.empty((4, 3))

            corners_position_3D[0] = np.array([0, 1.05, 1.05])
            corners_position_3D[1] = np.array([0, -1.05, 1.05])
            corners_position_3D[2] = np.array([0, -1.05, -1.05])
            corners_position_3D[3] = np.array([0, 1.05, -1.05])

            corners_position_3D = np.matmul(corners_position_3D, gate_rotation.T)
            corners_position_3D += gate_position_3D

            # Drone position and orientation
            drone_position_3D = np.array([[dataset['drone_x'][i], dataset['drone_y'][i], dataset['drone_z'][i]]])
            drone_orientation = np.array([[-dataset['drone_r'][i], -dataset['drone_p'][i], -dataset['drone_yaw'][i]]])
            drone_orientation += self.camera_rotation_offset

            # Rotation matrix
            Rx = np.array([[1, 0, 0], 
                            [0, np.cos(drone_orientation[0][0]), -np.sin(drone_orientation[0][0])],
                            [0, np.sin(drone_orientation[0][0]), np.cos(drone_orientation[0][0])]])

            Ry = np.array([[np.cos(drone_orientation[0][1]), 0, np.sin(drone_orientation[0][1])], 
                            [0, 1, 0],
                            [-np.sin(drone_orientation[0][1]), 0, np.cos(drone_orientation[0][1])]])

            Rz = np.array([[np.cos(drone_orientation[0][2]), -np.sin(drone_orientation[0][2]), 0], 
                            [np.sin(drone_orientation[0][2]), np.cos(drone_orientation[0][2]), 0],
                            [0, 0, 1]])

            # Drone rotation
            drone_rotation = np.matmul(np.matmul(Rx,Ry), Rz)

            # Transforming gate position into drone's body position
            gate_position_relative_3D = np.matmul((gate_position_3D - drone_position_3D) , drone_rotation.T) - self.camera_position_offset
            gate_position_camera = np.array([[-gate_position_relative_3D[0][1], -gate_position_relative_3D[0][2], gate_position_relative_3D[0][0], 1]])
            gate_position_2D = np.matmul(gate_position_camera, self.projection_matrix.T)
            gate_position_2D = gate_position_2D / gate_position_2D[0][2]

            # Transforming gate corners' position into drone's body position
            corners_position_relative_3D = np.matmul(corners_position_3D - np.tile(drone_position_3D, (4, 1)), drone_rotation.T) - self.camera_position_offset
            corners_position_camera = np.vstack((-corners_position_relative_3D[:,1], -corners_position_relative_3D[:,2], corners_position_relative_3D[:,0], np.ones(4,))).T
            corners_position_2D = np.matmul(corners_position_camera, self.projection_matrix.T)
            corners_position_2D[:,0] = np.divide(corners_position_2D[:,0], corners_position_2D[:,2])
            corners_position_2D[:,1] = np.divide(corners_position_2D[:,1], corners_position_2D[:,2])
            corners_position_2D = corners_position_2D[:,:2]

            # Calculate top-left and bottom-right corners.
            xmin = int(np.min(corners_position_2D[:,0]))
            ymin = int(np.min(corners_position_2D[:,1]))
            xmax = int(np.max(corners_position_2D[:,0]))
            ymax = int(np.max(corners_position_2D[:,1]))

            # Get distance and orientation from Huy's folder.
            #d = np.abs(dataset['distance_relative'][i])
            x_relative = dataset['gate_x'][i]-dataset['drone_x'][i]
            y_relative = dataset['gate_y'][i]-dataset['drone_y'][i]
            z_relative = dataset['gate_z'][i]-dataset['drone_z'][i]


            yaw_relative = dataset['gate_yaw'][i]-np.pi-dataset['drone_yaw'][i]

            # Create output folder.
            if not os.path.isdir(dest_folder):
                print("%s created." % dest_folder)
                os.mkdir(dest_folder)
                os.mkdir(os.path.join(dest_folder, 'images'))

            # Get image name and open image to learn dimension.
            # TODO is there a way to read dimensions from image header instead of reading whole image?
            image_name = dataset['code_name'][i]+'.png'

            img = cv2.imread(os.path.join(huy_folder_name, image_name))
            if img is None:
                print("[ WARNING ] Image does not exist in folder: {}".format(os.path.join(huy_folder_name, image_name)))
                continue

            """
            # If all corners are in the image plane.
            #   put sample into the dataset.        
            if xmin>0 and ymin>0 and xmax<img.shape[1] and ymax<img.shape[0]:

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)
                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                copyfile(os.path.join(huy_folder_name, image_name), os.path.join(dest_folder,'images', image_name))

            else:
                print("[*] {} skipped: {} {} {} {}.".format(image_name, xmin, ymin, xmax, ymax))
            """

            # If the center is in the image plane.
            #   put sample into the dataset.        
            center_x = int((xmin+xmax)/2)
            center_y = int((ymin+ymax)/2)
            if center_x>0 and center_y>0 and center_x<img.shape[1] and center_y<img.shape[0]:

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)
                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                copyfile(os.path.join(huy_folder_name, image_name), os.path.join(dest_folder,'images', image_name))

                cv2.putText(img, "gate yaw: {:.2f} ({:.1f})".format(dataset['gate_yaw'][i], 
                                            math.degrees(dataset['gate_yaw'][i])), (10, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "drone yaw: {:.2f} ({:.1f})".format(dataset['drone_yaw'][i], math.degrees(dataset['drone_yaw'][i])), 
                                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "yaw rel.: {:.2f} ({:.1f})".format(yaw_relative, math.degrees(yaw_relative)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                cv2.putText(img, "dist. : {:.2f}".format(distance)+ 'm.', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255),3 )
                cv2.imwrite("debug/"+image_name, img)

            else:
                print("[*] {} skipped: {} {} {} {}.".format(image_name, xmin, ymin, xmax, ymax))



        # Write annotations.
        with open(os.path.join(dest_folder, 'annotations.json'), 'w+') as f:
            json.dump(output, f)

        # Create and write train and test indices.
        a = np.arange(len(output["annotations"]))
        np.random.shuffle(a)
        num_sample = a.shape[0]

        train_indices = a[:int(num_sample*0.9)]
        test_indices = a[int(num_sample*0.9):]

        np.save(os.path.join(dest_folder, "train-indices.npy"), train_indices)
        np.save(os.path.join(dest_folder, "test-indices.npy"), test_indices)

           
h2d = HUY2DATASET()


"""
h2d.run('/media/my_folder/single_gate_intel_drone_16_Jan/batch_1', 'DATASETS/single_gate_intel_drone_16_Jan_batch_1')
h2d.run('/media/my_folder/single_gate_intel_drone_16_Jan/batch_2', 'DATASETS/single_gate_intel_drone_16_Jan_batch_2')
h2d.run('/media/my_folder/single_gate_intel_drone_16_Jan/batch_3', 'DATASETS/single_gate_intel_drone_16_Jan_batch_3')
h2d.run('/media/my_folder/single_gate_intel_drone_16_Jan/batch_4', 'DATASETS/single_gate_intel_drone_16_Jan_batch_4')
h2d.run('/media/my_folder/single_gate_intel_drone_16_Jan/batch_5', 'DATASETS/single_gate_intel_drone_16_Jan_batch_5')
h2d.run('/media/my_folder/single_gate_intel_drone_16_Jan/batch_6', 'DATASETS/single_gate_intel_drone_16_Jan_batch_6')
h2d.run('/media/my_folder/single_gate_intel_drone_16_Jan/batch_7', 'DATASETS/single_gate_intel_drone_16_Jan_batch_7')
h2d.run('/media/my_folder/single_gate_intel_drone_16_Jan/batch_8', 'DATASETS/single_gate_intel_drone_16_Jan_batch_8')
"""
"""
h2d.run('/media/my_folder/single_gate_intel_drone_11_Feb/batch_1', 'DATASETS/single_gate_intel_drone_11_Feb_batch_1')
h2d.run('/media/my_folder/single_gate_intel_drone_11_Feb/batch_2', 'DATASETS/single_gate_intel_drone_11_Feb_batch_2')
h2d.run('/media/my_folder/single_gate_intel_drone_11_Feb/batch_3', 'DATASETS/single_gate_intel_drone_11_Feb_batch_3')
h2d.run('/media/my_folder/single_gate_intel_drone_11_Feb/batch_4', 'DATASETS/single_gate_intel_drone_11_Feb_batch_4')
h2d.run('/media/my_folder/single_gate_intel_drone_11_Feb/batch_5', 'DATASETS/single_gate_intel_drone_11_Feb_batch_5')
h2d.run('/media/my_folder/single_gate_intel_drone_11_Feb/batch_6', 'DATASETS/single_gate_intel_drone_11_Feb_batch_6')
"""



def create_split_files(path_to_dataset, train_split=0.9, test_split=0.1):
    print("[*] dataset: {}".format(path_to_dataset))
    print("[*] train-test split: {}, {}".format(train_split, test_split))

    image_names = [f for f in os.listdir(os.path.join(path_to_dataset, 'images')) if os.path.isfile(os.path.join(path_to_dataset, 'images', f))]
    num_samples = len(image_names)
    print("[*] {} samples found.".format(num_samples))

    assert(train_split+test_split==1.)
    a = np.arange(num_samples)
    np.random.shuffle(a)

    train_indices = a[:int(num_samples*train_split)]
    test_indices = a[int(num_samples*train_split):]
    
    np.save(os.path.join(path_to_dataset, "train-indices.npy"), train_indices)
    np.save(os.path.join(path_to_dataset, "test-indices.npy"), test_indices)
    print(os.path.join(path_to_dataset, "train-indices.npy"))
    print(os.path.join(path_to_dataset, "test-indices.npy"))

#create_split_files("DATASETS/merged-data-8July-large-annot-cond")    
#create_split_files("DATASETS/merged-data-8July-large-annot-cond-and-dark-8Sep")    
#create_split_files("DATASETS/au_dr")   

# create_split_files("DATASETS/New_Night_Aug09_merged/")


class HUY:
    def __init__(self):
        pass

    def __init__(self):
        # Create the projection matrix
        self.projection_matrix = np.array([[628.559937, 0.000000, 322.631874, 0.000000, 0.000000, 641.147705, 260.018180, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]]).T 
        self.projection_matrix = np.reshape(self.projection_matrix, (4,3), order='F').T

        # Estimated offsed between camera and UAV's COM
        self.camera_position_offset = np.array([0.05, -0.05, 0.02]) # in UAV body frame
        self.camera_rotation_offset = np.array([1.5, -1.9, -2.3]) / 180. * np.pi

    def display(self, huy_folder_name, index):

        print("[*] Huy folder: %s" % huy_folder_name)

        # Read label.txt file to a dictionary.   
        dataset = {} 
        with open(os.path.join(huy_folder_name, 'label.txt'), 'r') as label_file:
            label_names = label_file.readline().split(' ')
            label_names[-1] = label_names[-1][:-1]
            for name in label_names:
                dataset[name]=[]

            print("[*] Headers: %s" % str(label_names))
            for line in label_file.readlines():
                values = (line.split(' '))   
                values[-1] = values[-1][:-1]
                values = [values[0]] + [float(value) for value in values[1:]]
                for i in range(len(values)):
                    dataset[list(dataset.keys())[i]].append(values[i])
                
            
        num_of_samples = len(dataset[list(dataset.keys())[0]])
        print("[*] Num of samples found: %d" % num_of_samples)


        output = {"annotations":[]}
        for i in [index]:

            # Gate postion in Vicon frame
            gate_position_3D = np.array([[dataset['gate_x'][i], dataset['gate_y'][i], dataset['gate_z'][i]]])
            gate_orientation = np.array([[dataset['gate_r'][i], dataset['gate_p'][i], dataset['gate_yaw'][i]]])
            

            
            gate_rotation = np.array([[np.cos(gate_orientation[0,2]), -np.sin(gate_orientation[0,2]), 0],
                                    [np.sin(gate_orientation[0,2]), np.cos(gate_orientation[0,2]), 0],
                                    [0, 0, 1]])

            # Gate corners postion in Vicon frame
            corners_position_3D = np.empty((4, 3))

            corners_position_3D[0] = np.array([0, 1.05, 1.05])
            corners_position_3D[1] = np.array([0, -1.05, 1.05])
            corners_position_3D[2] = np.array([0, -1.05, -1.05])
            corners_position_3D[3] = np.array([0, 1.05, -1.05])

            corners_position_3D = np.matmul(corners_position_3D, gate_rotation.T)
            corners_position_3D += gate_position_3D

            # Drone position and orientation
            drone_position_3D = np.array([[dataset['drone_x'][i], dataset['drone_y'][i], dataset['drone_z'][i]]])
            drone_orientation = np.array([[-dataset['drone_r'][i], -dataset['drone_p'][i], -dataset['drone_yaw'][i]]])
            drone_orientation += self.camera_rotation_offset

            # Rotation matrix
            Rx = np.array([[1, 0, 0], 
                            [0, np.cos(drone_orientation[0][0]), -np.sin(drone_orientation[0][0])],
                            [0, np.sin(drone_orientation[0][0]), np.cos(drone_orientation[0][0])]])

            Ry = np.array([[np.cos(drone_orientation[0][1]), 0, np.sin(drone_orientation[0][1])], 
                            [0, 1, 0],
                            [-np.sin(drone_orientation[0][1]), 0, np.cos(drone_orientation[0][1])]])

            Rz = np.array([[np.cos(drone_orientation[0][2]), -np.sin(drone_orientation[0][2]), 0], 
                            [np.sin(drone_orientation[0][2]), np.cos(drone_orientation[0][2]), 0],
                            [0, 0, 1]])

            # Drone rotation
            drone_rotation = np.matmul(np.matmul(Rx,Ry), Rz)

            # Transforming gate position into drone's body position
            gate_position_relative_3D = np.matmul((gate_position_3D - drone_position_3D) , drone_rotation.T) - self.camera_position_offset
            gate_position_camera = np.array([[-gate_position_relative_3D[0][1], -gate_position_relative_3D[0][2], gate_position_relative_3D[0][0], 1]])
            gate_position_2D = np.matmul(gate_position_camera, self.projection_matrix.T)
            gate_position_2D = gate_position_2D / gate_position_2D[0][2]

            # Transforming gate corners' position into drone's body position
            corners_position_relative_3D = np.matmul(corners_position_3D - np.tile(drone_position_3D, (4, 1)), drone_rotation.T) - self.camera_position_offset
            corners_position_camera = np.vstack((-corners_position_relative_3D[:,1], -corners_position_relative_3D[:,2], corners_position_relative_3D[:,0], np.ones(4,))).T
            corners_position_2D = np.matmul(corners_position_camera, self.projection_matrix.T)
            corners_position_2D[:,0] = np.divide(corners_position_2D[:,0], corners_position_2D[:,2])
            corners_position_2D[:,1] = np.divide(corners_position_2D[:,1], corners_position_2D[:,2])
            corners_position_2D = corners_position_2D[:,:2]

            # Calculate top-left and bottom-right corners.
            xmin = int(np.min(corners_position_2D[:,0]))
            ymin = int(np.min(corners_position_2D[:,1]))
            xmax = int(np.max(corners_position_2D[:,0]))
            ymax = int(np.max(corners_position_2D[:,1]))

            # Get distance and orientation from Huy's folder.
            #d = np.abs(dataset['distance_relative'][i])
            x_relative = dataset['gate_x'][i]-dataset['drone_x'][i]
            y_relative = dataset['gate_y'][i]-dataset['drone_y'][i]
            z_relative = dataset['gate_z'][i]-dataset['drone_z'][i]


            yaw_relative = dataset['gate_yaw'][i]-dataset['drone_yaw'][i]


            

            # Get image name and open image to learn dimension.
            # TODO is there a way to read dimensions from image header instead of reading whole image?
            image_name = dataset['code_name'][i]+'.png'

            img = cv2.imread(os.path.join(huy_folder_name, image_name))
            if img is None:
                print("[ WARNING ] Image does not exist in folder: {}".format(os.path.join(huy_folder_name, image_name)))
                continue

            # If center is in the image plane.
            #   put sample into the dataset.
            center_x = int((xmin+xmax)/2)
            center_y = int((ymin+ymax)/2)
            if center_x>0 and center_y>0 and center_x<img.shape[1] and center_y<img.shape[0]:

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)

                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                

                cv2.putText(img, "gate yaw: {:.2f} ({:.1f})".format(dataset['gate_yaw'][i], 
                                            math.degrees(dataset['gate_yaw'][i])), (10, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "drone yaw: {:.2f} ({:.1f})".format(dataset['drone_yaw'][i], math.degrees(dataset['drone_yaw'][i])), 
                                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "yaw rel.: {:.2f} ({:.1f})".format(yaw_relative, math.degrees(yaw_relative)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                cv2.putText(img, "dist. : {:.2f}".format(distance)+ 'm.', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255),3 )
                cv2.imwrite(image_name, img)
                print("[*] Distance: {}, Yaw relative in txt: {}, yaw relative calculated: {}".format(distance, dataset['yaw_relative'][i], yaw_relative))
                print("[*] Drone yaw: {}, gate yaw {}".format(dataset['drone_yaw'][i], dataset['gate_yaw'][i]))


            else:
                print("[*] {} skipped: {} {}.".format(image_name, center_x, center_y))


            """
            # If all corners are in the image plane.
            #   put sample into the dataset.
            if xmin>0 and ymin>0 and xmax<img.shape[1] and ymax<img.shape[0]:

                #mod1 = (dataset['gate_yaw'][i]-np.pi)%(np.pi*2)
                #mod2 = dataset['drone_yaw'][i]%(np.pi*2)

                #disp = mod1-mod2
                #if disp > np.pi:
                #    disp=-(2*np.pi-disp)

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)

                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                

                cv2.putText(img, "gate yaw: {:.2f} ({:.1f})".format(dataset['gate_yaw'][i], 
                                            math.degrees(dataset['gate_yaw'][i])), (10, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "drone yaw: {:.2f} ({:.1f})".format(dataset['drone_yaw'][i], math.degrees(dataset['drone_yaw'][i])), 
                                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "yaw rel.: {:.2f} ({:.1f})".format(yaw_relative, math.degrees(yaw_relative)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                cv2.putText(img, "dist. : {:.2f}".format(distance)+ 'm.', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255),3 )
                cv2.imwrite(image_name, img)
                print("[*] Distance: {}, Yaw relative in txt: {}, yaw relative calculated: {}".format(distance, dataset['yaw_relative'][i], yaw_relative))
                print("[*] Drone yaw: {}, gate yaw {}".format(dataset['drone_yaw'][i], dataset['gate_yaw'][i]))

            else:
                print("[*] {} skipped: {} {} {} {}.".format(image_name, xmin, ymin, xmax, ymax))
            
            #exit()
            """
"""
huy = HUY()

for i in [0,300,500]:
    huy.display('RAW_DATA/data1_jan16_tol_002/', i)
    huy.display('RAW_DATA/data2_jan16_tol_002/', i)
    huy.display('RAW_DATA/data3_jan16_tol_002/', i)
    huy.display('RAW_DATA/data4_jan16_tol_002/', i)
    huy.display('RAW_DATA/data5_jan16_tol_002/', i)
    huy.display('RAW_DATA/data6_jan16_tol_002/', i)
    huy.display('RAW_DATA/data7_jan16_tol_002/', i)
    huy.display('RAW_DATA/data8_jan16_tol_002/', i)

    huy.display('RAW_DATA/feb_data_1/', i)
    huy.display('RAW_DATA/feb_data_2/', i)
    huy.display('RAW_DATA/feb_data_3/', i)
    huy.display('RAW_DATA/feb_data_4/', i)
    huy.display('RAW_DATA/feb_data_5/', i)
    huy.display('RAW_DATA/feb_data_6/', i)
"""



class HUYFishEye2DATASET:
    """ Create labels for training using back projection. 
        Use vicon data for this.

        This class is a converter to 3D coordinates of gates to 2D pixel coordinates.
    """

    def __init__(self):

        # From camera calibration .yaml file
        self.distortion_coeffs = np.array([0.008242625132650467, -0.0038525210246120227, 0.004515885224263602, -0.0014883484550862235])
        self.intrinsics = np.array([636.6532939717728, 636.8863079330886, 718.9654948793151, 538.6638261352892])
        self.resolution = [1440, 1080]

        # Camera matrix
        camera_matrix = np.array([intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1]).reshape((3,3))

        # Estimated offsed between camera and UAV's COM
        self.camera_position_offset = np.array([0.05, 0.00, 0.01]) # in UAV body frame
        self.camera_rotation_offset = np.array([-1, 22, -3]) / 180. * np.pi

import json
import math

#PATH_TO_DATASET = "/gate_pose_dataset/"

def read_annot(path_to_annot_file):
    with open(path_to_annot_file, 'r') as json_file:
        data = json.load(json_file)
        print(data.keys())
        print(data['classes'])
        for i in range(10):
            print(data['annotations'][i])
      

#read_annot(PATH_TO_DATASET+'training/'+'lights_hand_flight/annotations.json')        


import cv2
import os
import json
import shutil
import numpy as np
from shutil import copyfile


def recursive(folder):
    paths = []
    
    dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for d in dirs:
        path_dir = os.path.join(folder, d)
        if os.path.isfile(os.path.join(path_dir, "annotations.json")):
            paths.append(path_dir)

        else:
            paths += recursive(path_dir)

    return paths


def merge_hybrid_dataset(destination, source):

    if os.path.isdir(destination):
        print("[*] Destination folder: %s" % destination)
        if not os.path.isdir(os.path.join(destination, 'images')):
            os.mkdir(os.path.join(destination, 'images'))
    else:
        raise RuntimeError("Destination folder does not exist: %s" %destination)

    if os.path.isdir(source):
        print("[*] Source folder: %s" % source)
    else:
        raise RuntimeError("Source folder does not exist: %s" %source)

    cnt = 0

    data_folders = recursive(source)
    print("[*] Data folders: %s" % str(data_folders))


    annot_file = open(os.path.join(destination, "annotations.json"), "w+")
    content = {'annotations':[]}
    for data_folder in data_folders:

        print(os.path.join(data_folder, "annotations.json"))
        with open(os.path.join(data_folder, "annotations.json"), 'r') as json_file:
            data = json.load(json_file)

            for annot in data["annotations"]:

                img_name = annot["image"]
                img_path = os.path.join(data_folder, "images", img_name)
                print(img_path)

                annot['image']='{0:06d}'.format(cnt)+'.png'
                shutil.copyfile(img_path, os.path.join(destination, "images", annot['image']))

                content["annotations"].append(annot)
                cnt += 1
                #print(annot)            
    
    json.dump(content, annot_file)
    annot_file.close()

"""
destination = "DATASETS/17June-merged"
source = "DATASETS/17June"    

merge_hybrid_dataset(destination, source)
"""

"""
destination = "huy-9Jan-and-11Feb-merged-raw-orientation"
source = "huy-9Jan-and-11Feb-raw-orientation"    

merge_hybrid_dataset(destination, source)
"""


#destination = "DATASETS/New_Night_Aug09_merged"
#source = "DATASETS/New_Night_Aug09"    

#merge_hybrid_dataset(destination, source)



class HUY2DATASET:
    """ Create labels for training using back projection. 
        Use vicon data for this.

        This class is a converter to 3D coordinates of gates to 2D pixel coordinates.
    """

    def __init__(self):
        # Create the projection matrix
        self.projection_matrix = np.array([[628.559937, 0.000000, 322.631874, 0.000000, 0.000000, 641.147705, 260.018180, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]]).T 
        self.projection_matrix = np.reshape(self.projection_matrix, (4,3), order='F').T

        # Estimated offsed between camera and UAV's COM
        self.camera_position_offset = np.array([0.05, -0.05, 0.02]) # in UAV body frame
        self.camera_rotation_offset = np.array([1.5, -1.9, -2.3]) / 180. * np.pi

    def run(self, huy_folder_name, dest_folder):

        print("[*] Huy folder: %s" % huy_folder_name)

        # Read label.txt file to a dictionary.   
        dataset = {} 
        with open(os.path.join(huy_folder_name, 'label.txt'), 'r') as label_file:
            label_names = label_file.readline().split(' ')
            label_names[-1] = label_names[-1][:-1]
            for name in label_names:
                dataset[name]=[]

            print("[*] Headers: %s" % str(label_names))
            for line in label_file.readlines():
                values = (line.split(' '))   
                values[-1] = values[-1][:-1]
                values = [values[0]] + [float(value) for value in values[1:]]
                for i in range(len(values)):
                    dataset[list(dataset.keys())[i]].append(values[i])
                
            
        num_of_samples = len(dataset[list(dataset.keys())[0]])
        print("[*] Num of samples found: %d" % num_of_samples)


        output = {"annotations":[]}
        for i in range(num_of_samples):

            # Gate postion in Vicon frame
            gate_position_3D = np.array([[dataset['gate_x'][i], dataset['gate_y'][i], dataset['gate_z'][i]]])
            gate_orientation = np.array([[dataset['gate_r'][i], dataset['gate_p'][i], dataset['gate_yaw'][i]]])
            

            
            gate_rotation = np.array([[np.cos(gate_orientation[0,2]), -np.sin(gate_orientation[0,2]), 0],
                                    [np.sin(gate_orientation[0,2]), np.cos(gate_orientation[0,2]), 0],
                                    [0, 0, 1]])

            # Gate corners postion in Vicon frame
            corners_position_3D = np.empty((4, 3))

            corners_position_3D[0] = np.array([0, 1.05, 1.05])
            corners_position_3D[1] = np.array([0, -1.05, 1.05])
            corners_position_3D[2] = np.array([0, -1.05, -1.05])
            corners_position_3D[3] = np.array([0, 1.05, -1.05])

            corners_position_3D = np.matmul(corners_position_3D, gate_rotation.T)
            corners_position_3D += gate_position_3D

            # Drone position and orientation
            drone_position_3D = np.array([[dataset['drone_x'][i], dataset['drone_y'][i], dataset['drone_z'][i]]])
            drone_orientation = np.array([[-dataset['drone_r'][i], -dataset['drone_p'][i], -dataset['drone_yaw'][i]]])
            drone_orientation += self.camera_rotation_offset

            # Rotation matrix
            Rx = np.array([[1, 0, 0], 
                            [0, np.cos(drone_orientation[0][0]), -np.sin(drone_orientation[0][0])],
                            [0, np.sin(drone_orientation[0][0]), np.cos(drone_orientation[0][0])]])

            Ry = np.array([[np.cos(drone_orientation[0][1]), 0, np.sin(drone_orientation[0][1])], 
                            [0, 1, 0],
                            [-np.sin(drone_orientation[0][1]), 0, np.cos(drone_orientation[0][1])]])

            Rz = np.array([[np.cos(drone_orientation[0][2]), -np.sin(drone_orientation[0][2]), 0], 
                            [np.sin(drone_orientation[0][2]), np.cos(drone_orientation[0][2]), 0],
                            [0, 0, 1]])

            # Drone rotation
            drone_rotation = np.matmul(np.matmul(Rx,Ry), Rz)

            # Transforming gate position into drone's body position
            gate_position_relative_3D = np.matmul((gate_position_3D - drone_position_3D) , drone_rotation.T) - self.camera_position_offset
            gate_position_camera = np.array([[-gate_position_relative_3D[0][1], -gate_position_relative_3D[0][2], gate_position_relative_3D[0][0], 1]])
            gate_position_2D = np.matmul(gate_position_camera, self.projection_matrix.T)
            gate_position_2D = gate_position_2D / gate_position_2D[0][2]

            # Transforming gate corners' position into drone's body position
            corners_position_relative_3D = np.matmul(corners_position_3D - np.tile(drone_position_3D, (4, 1)), drone_rotation.T) - self.camera_position_offset
            corners_position_camera = np.vstack((-corners_position_relative_3D[:,1], -corners_position_relative_3D[:,2], corners_position_relative_3D[:,0], np.ones(4,))).T
            corners_position_2D = np.matmul(corners_position_camera, self.projection_matrix.T)
            corners_position_2D[:,0] = np.divide(corners_position_2D[:,0], corners_position_2D[:,2])
            corners_position_2D[:,1] = np.divide(corners_position_2D[:,1], corners_position_2D[:,2])
            corners_position_2D = corners_position_2D[:,:2]

            # Calculate top-left and bottom-right corners.
            xmin = int(np.min(corners_position_2D[:,0]))
            ymin = int(np.min(corners_position_2D[:,1]))
            xmax = int(np.max(corners_position_2D[:,0]))
            ymax = int(np.max(corners_position_2D[:,1]))

            # Get distance and orientation from Huy's folder.
            #d = np.abs(dataset['distance_relative'][i])
            x_relative = dataset['gate_x'][i]-dataset['drone_x'][i]
            y_relative = dataset['gate_y'][i]-dataset['drone_y'][i]
            z_relative = dataset['gate_z'][i]-dataset['drone_z'][i]


            yaw_relative = dataset['gate_yaw'][i]-dataset['drone_yaw'][i]

            # Create output folder.
            if not os.path.isdir(dest_folder):
                print("%s created." % dest_folder)
                os.mkdir(dest_folder)
                os.mkdir(os.path.join(dest_folder, 'images'))

            # Get image name and open image to learn dimension.
            # TODO is there a way to read dimensions from image header instead of reading whole image?
            image_name = dataset['code_name'][i]+'.png'

            img = cv2.imread(os.path.join(huy_folder_name, image_name))
            if img is None:
                print("[ WARNING ] Image does not exist in folder: {}".format(os.path.join(huy_folder_name, image_name)))
                continue

            """
            # If all corners are in the image plane.
            #   put sample into the dataset.        
            if xmin>0 and ymin>0 and xmax<img.shape[1] and ymax<img.shape[0]:

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)
                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                copyfile(os.path.join(huy_folder_name, image_name), os.path.join(dest_folder,'images', image_name))

            else:
                print("[*] {} skipped: {} {} {} {}.".format(image_name, xmin, ymin, xmax, ymax))
            """

            # If the center is in the image plane.
            #   put sample into the dataset.        
            center_x = int((xmin+xmax)/2)
            center_y = int((ymin+ymax)/2)
            if center_x>0 and center_y>0 and center_x<img.shape[1] and center_y<img.shape[0]:

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)
                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                copyfile(os.path.join(huy_folder_name, image_name), os.path.join(dest_folder,'images', image_name))

            else:
                print("[*] {} skipped: {} {} {} {}.".format(image_name, xmin, ymin, xmax, ymax))



        # Write annotations.
        with open(os.path.join(dest_folder, 'annotations.json'), 'w+') as f:
            json.dump(output, f)

        # Create and write train and test indices.
        a = np.arange(len(output["annotations"]))
        np.random.shuffle(a)
        num_sample = a.shape[0]

        train_indices = a[:int(num_sample*0.9)]
        test_indices = a[int(num_sample*0.9):]

        np.save(os.path.join(dest_folder, "train-indices.npy"), train_indices)
        np.save(os.path.join(dest_folder, "test-indices.npy"), test_indices)

"""            
h2d = HUY2DATASET()        
h2d.run('RAW_DATA/data1_jan16_tol_002/', 'huy1')
h2d.run('RAW_DATA/data2_jan16_tol_002/', 'huy2')
h2d.run('RAW_DATA/data3_jan16_tol_002/', 'huy3')
h2d.run('RAW_DATA/data4_jan16_tol_002/', 'huy4')
h2d.run('RAW_DATA/data5_jan16_tol_002/', 'huy5')
h2d.run('RAW_DATA/data6_jan16_tol_002/', 'huy6')
h2d.run('RAW_DATA/data7_jan16_tol_002/', 'huy7')
h2d.run('RAW_DATA/data8_jan16_tol_002/', 'huy8')
h2d.run('RAW_DATA/feb_data_1/', 'huy9')
h2d.run('RAW_DATA/feb_data_2/', 'huy10')
h2d.run('RAW_DATA/feb_data_3/', 'huy11')
h2d.run('RAW_DATA/feb_data_4/', 'huy12')
h2d.run('RAW_DATA/feb_data_5/', 'huy13')
h2d.run('RAW_DATA/feb_data_6/', 'huy14')
"""



def create_split_files(path_to_dataset, train_split=0.9, test_split=0.1):
    print("[*] dataset: {}".format(path_to_dataset))
    print("[*] train-test split: {}, {}".format(train_split, test_split))

    image_names = [f for f in os.listdir(os.path.join(path_to_dataset, 'images')) if os.path.isfile(os.path.join(path_to_dataset, 'images', f))]
    num_samples = len(image_names)
    print("[*] {} samples found.".format(num_samples))

    assert(train_split+test_split==1.)
    a = np.arange(num_samples)
    np.random.shuffle(a)

    train_indices = a[:int(num_samples*train_split)]
    test_indices = a[int(num_samples*train_split):]
    
    np.save(os.path.join(path_to_dataset, "train-indices.npy"), train_indices)
    np.save(os.path.join(path_to_dataset, "test-indices.npy"), test_indices)
    

#create_split_files("DATASETS/17June-merged")    
#create_split_files("DATASETS/night-dataset-1-3", train_split=0.95, test_split=0.05)

class HUY:
    def __init__(self):
        pass

    def __init__(self):
        # Create the projection matrix
        self.projection_matrix = np.array([[628.559937, 0.000000, 322.631874, 0.000000, 0.000000, 641.147705, 260.018180, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]]).T 
        self.projection_matrix = np.reshape(self.projection_matrix, (4,3), order='F').T

        # Estimated offsed between camera and UAV's COM
        self.camera_position_offset = np.array([0.05, -0.05, 0.02]) # in UAV body frame
        self.camera_rotation_offset = np.array([1.5, -1.9, -2.3]) / 180. * np.pi

    def display(self, huy_folder_name, index):

        print("[*] Huy folder: %s" % huy_folder_name)

        # Read label.txt file to a dictionary.   
        dataset = {} 
        with open(os.path.join(huy_folder_name, 'label.txt'), 'r') as label_file:
            label_names = label_file.readline().split(' ')
            label_names[-1] = label_names[-1][:-1]
            for name in label_names:
                dataset[name]=[]

            print("[*] Headers: %s" % str(label_names))
            for line in label_file.readlines():
                values = (line.split(' '))   
                values[-1] = values[-1][:-1]
                values = [values[0]] + [float(value) for value in values[1:]]
                for i in range(len(values)):
                    dataset[list(dataset.keys())[i]].append(values[i])
                
            
        num_of_samples = len(dataset[list(dataset.keys())[0]])
        print("[*] Num of samples found: %d" % num_of_samples)


        output = {"annotations":[]}
        for i in [index]:

            # Gate postion in Vicon frame
            gate_position_3D = np.array([[dataset['gate_x'][i], dataset['gate_y'][i], dataset['gate_z'][i]]])
            gate_orientation = np.array([[dataset['gate_r'][i], dataset['gate_p'][i], dataset['gate_yaw'][i]]])
            

            
            gate_rotation = np.array([[np.cos(gate_orientation[0,2]), -np.sin(gate_orientation[0,2]), 0],
                                    [np.sin(gate_orientation[0,2]), np.cos(gate_orientation[0,2]), 0],
                                    [0, 0, 1]])

            # Gate corners postion in Vicon frame
            corners_position_3D = np.empty((4, 3))

            corners_position_3D[0] = np.array([0, 1.05, 1.05])
            corners_position_3D[1] = np.array([0, -1.05, 1.05])
            corners_position_3D[2] = np.array([0, -1.05, -1.05])
            corners_position_3D[3] = np.array([0, 1.05, -1.05])

            corners_position_3D = np.matmul(corners_position_3D, gate_rotation.T)
            corners_position_3D += gate_position_3D

            # Drone position and orientation
            drone_position_3D = np.array([[dataset['drone_x'][i], dataset['drone_y'][i], dataset['drone_z'][i]]])
            drone_orientation = np.array([[-dataset['drone_r'][i], -dataset['drone_p'][i], -dataset['drone_yaw'][i]]])
            drone_orientation += self.camera_rotation_offset

            # Rotation matrix
            Rx = np.array([[1, 0, 0], 
                            [0, np.cos(drone_orientation[0][0]), -np.sin(drone_orientation[0][0])],
                            [0, np.sin(drone_orientation[0][0]), np.cos(drone_orientation[0][0])]])

            Ry = np.array([[np.cos(drone_orientation[0][1]), 0, np.sin(drone_orientation[0][1])], 
                            [0, 1, 0],
                            [-np.sin(drone_orientation[0][1]), 0, np.cos(drone_orientation[0][1])]])

            Rz = np.array([[np.cos(drone_orientation[0][2]), -np.sin(drone_orientation[0][2]), 0], 
                            [np.sin(drone_orientation[0][2]), np.cos(drone_orientation[0][2]), 0],
                            [0, 0, 1]])

            # Drone rotation
            drone_rotation = np.matmul(np.matmul(Rx,Ry), Rz)

            # Transforming gate position into drone's body position
            gate_position_relative_3D = np.matmul((gate_position_3D - drone_position_3D) , drone_rotation.T) - self.camera_position_offset
            gate_position_camera = np.array([[-gate_position_relative_3D[0][1], -gate_position_relative_3D[0][2], gate_position_relative_3D[0][0], 1]])
            gate_position_2D = np.matmul(gate_position_camera, self.projection_matrix.T)
            gate_position_2D = gate_position_2D / gate_position_2D[0][2]

            # Transforming gate corners' position into drone's body position
            corners_position_relative_3D = np.matmul(corners_position_3D - np.tile(drone_position_3D, (4, 1)), drone_rotation.T) - self.camera_position_offset
            corners_position_camera = np.vstack((-corners_position_relative_3D[:,1], -corners_position_relative_3D[:,2], corners_position_relative_3D[:,0], np.ones(4,))).T
            corners_position_2D = np.matmul(corners_position_camera, self.projection_matrix.T)
            corners_position_2D[:,0] = np.divide(corners_position_2D[:,0], corners_position_2D[:,2])
            corners_position_2D[:,1] = np.divide(corners_position_2D[:,1], corners_position_2D[:,2])
            corners_position_2D = corners_position_2D[:,:2]

            # Calculate top-left and bottom-right corners.
            xmin = int(np.min(corners_position_2D[:,0]))
            ymin = int(np.min(corners_position_2D[:,1]))
            xmax = int(np.max(corners_position_2D[:,0]))
            ymax = int(np.max(corners_position_2D[:,1]))

            # Get distance and orientation from Huy's folder.
            #d = np.abs(dataset['distance_relative'][i])
            x_relative = dataset['gate_x'][i]-dataset['drone_x'][i]
            y_relative = dataset['gate_y'][i]-dataset['drone_y'][i]
            z_relative = dataset['gate_z'][i]-dataset['drone_z'][i]


            yaw_relative = dataset['gate_yaw'][i]-dataset['drone_yaw'][i]


            

            # Get image name and open image to learn dimension.
            # TODO is there a way to read dimensions from image header instead of reading whole image?
            image_name = dataset['code_name'][i]+'.png'

            img = cv2.imread(os.path.join(huy_folder_name, image_name))
            if img is None:
                print("[ WARNING ] Image does not exist in folder: {}".format(os.path.join(huy_folder_name, image_name)))
                continue

            # If center is in the image plane.
            #   put sample into the dataset.
            center_x = int((xmin+xmax)/2)
            center_y = int((ymin+ymax)/2)
            if center_x>0 and center_y>0 and center_x<img.shape[1] and center_y<img.shape[0]:

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)

                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                

                cv2.putText(img, "gate yaw: {:.2f} ({:.1f})".format(dataset['gate_yaw'][i], 
                                            math.degrees(dataset['gate_yaw'][i])), (10, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "drone yaw: {:.2f} ({:.1f})".format(dataset['drone_yaw'][i], math.degrees(dataset['drone_yaw'][i])), 
                                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "yaw rel.: {:.2f} ({:.1f})".format(yaw_relative, math.degrees(yaw_relative)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                cv2.putText(img, "dist. : {:.2f}".format(distance)+ 'm.', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255),3 )
                cv2.imwrite(image_name, img)
                print("[*] Distance: {}, Yaw relative in txt: {}, yaw relative calculated: {}".format(distance, dataset['yaw_relative'][i], yaw_relative))
                print("[*] Drone yaw: {}, gate yaw {}".format(dataset['drone_yaw'][i], dataset['gate_yaw'][i]))


            else:
                print("[*] {} skipped: {} {}.".format(image_name, center_x, center_y))


            """
            # If all corners are in the image plane.
            #   put sample into the dataset.
            if xmin>0 and ymin>0 and xmax<img.shape[1] and ymax<img.shape[0]:

                #mod1 = (dataset['gate_yaw'][i]-np.pi)%(np.pi*2)
                #mod2 = dataset['drone_yaw'][i]%(np.pi*2)

                #disp = mod1-mod2
                #if disp > np.pi:
                #    disp=-(2*np.pi-disp)

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)

                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                

                cv2.putText(img, "gate yaw: {:.2f} ({:.1f})".format(dataset['gate_yaw'][i], 
                                            math.degrees(dataset['gate_yaw'][i])), (10, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "drone yaw: {:.2f} ({:.1f})".format(dataset['drone_yaw'][i], math.degrees(dataset['drone_yaw'][i])), 
                                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "yaw rel.: {:.2f} ({:.1f})".format(yaw_relative, math.degrees(yaw_relative)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                cv2.putText(img, "dist. : {:.2f}".format(distance)+ 'm.', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255),3 )
                cv2.imwrite(image_name, img)
                print("[*] Distance: {}, Yaw relative in txt: {}, yaw relative calculated: {}".format(distance, dataset['yaw_relative'][i], yaw_relative))
                print("[*] Drone yaw: {}, gate yaw {}".format(dataset['drone_yaw'][i], dataset['gate_yaw'][i]))

            else:
                print("[*] {} skipped: {} {} {} {}.".format(image_name, xmin, ymin, xmax, ymax))
            
            #exit()
            """
"""
huy = HUY()

for i in [0,300,500]:
    huy.display('RAW_DATA/data1_jan16_tol_002/', i)
    huy.display('RAW_DATA/data2_jan16_tol_002/', i)
    huy.display('RAW_DATA/data3_jan16_tol_002/', i)
    huy.display('RAW_DATA/data4_jan16_tol_002/', i)
    huy.display('RAW_DATA/data5_jan16_tol_002/', i)
    huy.display('RAW_DATA/data6_jan16_tol_002/', i)
    huy.display('RAW_DATA/data7_jan16_tol_002/', i)
    huy.display('RAW_DATA/data8_jan16_tol_002/', i)

    huy.display('RAW_DATA/feb_data_1/', i)
    huy.display('RAW_DATA/feb_data_2/', i)
    huy.display('RAW_DATA/feb_data_3/', i)
    huy.display('RAW_DATA/feb_data_4/', i)
    huy.display('RAW_DATA/feb_data_5/', i)
    huy.display('RAW_DATA/feb_data_6/', i)
"""



class HUYFishEye2DATASET:
    """ Create labels for training using back projection. 
        Use vicon data for this.

        This class is a converter to 3D coordinates of gates to 2D pixel coordinates.
    """

    def __init__(self):

        # From camera calibration .yaml file
        self.distortion_coeffs = np.array([0.008242625132650467, -0.0038525210246120227, 0.004515885224263602, -0.0014883484550862235])
        self.intrinsics = np.array([636.6532939717728, 636.8863079330886, 718.9654948793151, 538.6638261352892])
        self.resolution = [1440, 1080]

        # Camera matrix
        self.camera_matrix = np.array([self.intrinsics[0], 0, self.intrinsics[2], 0, self.intrinsics[1], self.intrinsics[3], 0, 0, 1]).reshape((3,3))

        # Estimated offsed between camera and UAV's COM
        self.camera_position_offset = np.array([-0.9, 0.00, -0.03]) # in UAV body frame
        self.camera_rotation_offset = np.array([6, 28, -3]) / 180. * np.pi




    def run(self, huy_folder_name, dest_folder, new_resolution=(512,384)):

        print("[*] Huy folder: %s" % huy_folder_name)

        # Read label.txt file to a dictionary.   
        dataset = {} 
        with open(os.path.join(huy_folder_name, 'label.txt'), 'r') as label_file:
            label_names = label_file.readline().split(' ')
            label_names[-1] = label_names[-1][:-1]
            for name in label_names:
                dataset[name]=[]

            print("[*] Headers: %s" % str(label_names))
            for line in label_file.readlines():
                if line[0]=="c":
                    print(line)
                    continue
                values = (line.split(' '))   
                values[-1] = values[-1][:-1]
                values = [values[0]] + [float(value) for value in values[1:]]
                for i in range(len(values)):
                    dataset[list(dataset.keys())[i]].append(values[i])
                
            
        num_of_samples = len(dataset[list(dataset.keys())[0]])
        print("[*] Num of samples found: %d" % num_of_samples)


        output = {"annotations":[]}
        for i in range(2, num_of_samples):

            # Gate postion in Vicon frame
            gate_position_3D = np.array([[dataset['gate_x'][i-2], dataset['gate_y'][i-2], dataset['gate_z'][i-2]]])
            gate_orientation = np.array([[dataset['gate_roll'][i-2], dataset['gate_pitch'][i-2], dataset['gate_yaw'][i-2]]])
            
            
            gate_rotation = np.array([[np.cos(gate_orientation[0,2]), -np.sin(gate_orientation[0,2]), 0],
                                    [np.sin(gate_orientation[0,2]), np.cos(gate_orientation[0,2]), 0],
                                    [0, 0, 1]])

            # Gate corners postion in Vicon frame
            corners_position_3D = np.empty((9, 3))

            corners_position_3D[0] = np.array([0, 1.05, 1.05])
            corners_position_3D[1] = np.array([0, 0, 1.05])
            corners_position_3D[2] = np.array([0, -1.05, 1.05])
            corners_position_3D[3] = np.array([0, -1.05, 0])
            corners_position_3D[4] = np.array([0, -1.05, -1.05])
            corners_position_3D[5] = np.array([0, 0, -1.05])
            corners_position_3D[6] = np.array([0, 1.05, -1.05])
            corners_position_3D[7] = np.array([0, 1.05, 0])
            corners_position_3D[8] = np.array([0, 1.05, 1.05])

            corners_position_3D = np.matmul(corners_position_3D, gate_rotation.T)
            corners_position_3D += gate_position_3D

            # Drone position and orientation
            drone_position_3D = np.array([[dataset['drone_x'][i-2], dataset['drone_y'][i-2], dataset['drone_z'][i-2]]])
            drone_orientation = np.array([[-dataset['drone_roll'][i-2], -dataset['drone_pitch'][i-2], -dataset['drone_yaw'][i-2]]])
            drone_orientation += self.camera_rotation_offset

            # Rotation matrix
            Rx = np.array([[1, 0, 0], 
                            [0, np.cos(drone_orientation[0][0]), -np.sin(drone_orientation[0][0])],
                            [0, np.sin(drone_orientation[0][0]), np.cos(drone_orientation[0][0])]])

            Ry = np.array([[np.cos(drone_orientation[0][1]), 0, np.sin(drone_orientation[0][1])], 
                            [0, 1, 0],
                            [-np.sin(drone_orientation[0][1]), 0, np.cos(drone_orientation[0][1])]])

            Rz = np.array([[np.cos(drone_orientation[0][2]), -np.sin(drone_orientation[0][2]), 0], 
                            [np.sin(drone_orientation[0][2]), np.cos(drone_orientation[0][2]), 0],
                            [0, 0, 1]])

            # Drone rotation
            # TODO: Is this correct order
            # drone_rotation = np.matmul(np.matmul(Rx,Ry), Rz)
            drone_rotation = np.matmul(np.matmul(Rx,Ry),Rz)


            # Transforming gate position into drone's body position
            gate_position_relative_3D = np.matmul((gate_position_3D - drone_position_3D) , drone_rotation.T) - self.camera_position_offset
            gate_position_camera = np.array([[-gate_position_relative_3D[0][1], -gate_position_relative_3D[0][2], gate_position_relative_3D[0][0]]])

            # Pinhole projection coordinates of gate's position in camera frame
            gate_projection = gate_position_camera/gate_position_camera[0][2]
            r = np.sqrt(gate_projection[0][0]**2 + gate_projection[0][1]**2);
            theta = np.arctan(r)

            # Fisheye distortion
            theta_d = np.matmul(np.hstack((np.array(1), self.distortion_coeffs)), np.array([[theta], [np.power(theta,3)], [np.power(theta,5)], [np.power(theta,7)], [np.power(theta,9)]]))

            # Distorted point coordinates
            gate_distorted = theta_d/r*gate_projection

            # The final pixel coordinates vector
            gate_distorted[:,2] = 1
            gate_position_2D = np.matmul(self.camera_matrix,gate_distorted.T).T          

            # Transforming gate corners' position into drone's body position
            corners_position_relative_3D = np.matmul(corners_position_3D - np.tile(drone_position_3D, (9, 1)), drone_rotation.T) - self.camera_position_offset            
            corners_position_camera = np.vstack((-corners_position_relative_3D[:,1], -corners_position_relative_3D[:,2], corners_position_relative_3D[:,0])).T
            
            # TODO is necessery?
            #corners_position_camera(corners_position_camera(:,3) < 0,:) = NaN; % removes the corners behind the camera
            
            # Pinhole projection coordinates of corners' position in camera frame
            corners_projection = corners_position_camera/corners_position_camera[:,2].reshape((9,1))
            
            # Fisheye distortion
            r = np.sqrt(np.power(corners_projection[:,0],2) + np.power(corners_projection[:,1],2))
            theta = np.arctan(r)
            theta_d = np.matmul(np.array([theta, np.power(theta,3), np.power(theta,5), np.power(theta,7), np.power(theta,9)]).T, np.hstack((np.array(1), self.distortion_coeffs)).T)

            # Distorted point coordinates
            corners_distorted = theta_d.reshape((9,1))/r.reshape((9,1))*corners_projection
            
            # The final pixel coordinates vector
            corners_distorted[:,2] = 1
            #print(self.camera_matrix.shape)
            #print(corners_distorted.shape)
            corners_position_2D = np.matmul(self.camera_matrix,corners_distorted.T).T
            #print(corners_position_2D)
            

            # Calculate top-left and bottom-right corners.
            xmin = int(np.min(corners_position_2D[:,0]))
            ymin = int(np.min(corners_position_2D[:,1]))
            xmax = int(np.max(corners_position_2D[:,0]))
            ymax = int(np.max(corners_position_2D[:,1]))

            # Rescale if necessery.
            xmin = int(xmin*(new_resolution[0]/self.resolution[0]))
            ymin = int(ymin*(new_resolution[1]/self.resolution[1]))
            xmax = int(xmax*(new_resolution[0]/self.resolution[0]))
            ymax = int(ymax*(new_resolution[1]/self.resolution[1]))


            # Get distance and orientation from Huy's folder.
            #d = np.abs(dataset['distance_relative'][i])
            x_relative = dataset['gate_x'][i-2]-dataset['drone_x'][i-2]
            y_relative = dataset['gate_y'][i-2]-dataset['drone_y'][i-2]
            z_relative = dataset['gate_z'][i-2]-dataset['drone_z'][i-2]


            yaw_relative = dataset['gate_yaw'][i-2]-dataset['drone_yaw'][i-2]

            # Create output folder.
            if not os.path.isdir(dest_folder):
                print("%s created." % dest_folder)
                os.mkdir(dest_folder)
                os.mkdir(os.path.join(dest_folder, 'images'))

            # Get image name and open image to learn dimension.
            # TODO is there a way to read dimensions from image header instead of reading whole image?
            image_name = dataset['code_name'][i]+'.png'

            img = cv2.imread(os.path.join(huy_folder_name, image_name))
            img = cv2.resize(img, new_resolution)
            if img is None:
                print("[ WARNING ] Image does not exist in folder: {}".format(os.path.join(huy_folder_name, image_name)))
                continue

            """
            # If all corners are in the image plane.
            #   put sample into the dataset.        
            if gate_position_camera[0][2]>0 and center_x>0 and center_y>0 and center_x<img.shape[1] and center_y<img.shape[0]:

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)
                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                copyfile(os.path.join(huy_folder_name, image_name), os.path.join(dest_folder,'images', image_name))

            else:
                print("[*] {} skipped: {} {} {} {}.".format(image_name, xmin, ymin, xmax, ymax))
            """

            # If the center is in the image plane.
            #   put sample into the dataset.        
            center_x = int((xmin+xmax)/2)
            center_y = int((ymin+ymax)/2)
            if gate_position_camera[0][2]>0 and center_x>0 and center_y>0 and center_x<img.shape[1] and center_y<img.shape[0]:

                distance = np.sqrt(x_relative**2 + y_relative**2 + z_relative**2)
                sample = {"image":image_name, 
                    "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "yaw_relative":yaw_relative, "distance":distance}]}
                output['annotations'].append(sample)
                #copyfile(os.path.join(huy_folder_name, image_name), os.path.join(dest_folder,'images', image_name))
                cv2.imwrite(os.path.join(dest_folder,'images', image_name), img)

                cv2.putText(img, "gate yaw: {:.2f} ({:.1f})".format(dataset['gate_yaw'][i], 
                                            math.degrees(dataset['gate_yaw'][i])), (10, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "drone yaw: {:.2f} ({:.1f})".format(dataset['drone_yaw'][i], math.degrees(dataset['drone_yaw'][i])), 
                                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    

                cv2.putText(img, "yaw rel.: {:.2f} ({:.1f})".format(yaw_relative, math.degrees(yaw_relative)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      
                cv2.putText(img, "dist. : {:.2f}".format(distance)+ 'm.', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)                      

                cv2.circle(img, (int((xmin+xmax)/2), int((ymin+ymax)/2)), 5, (0,0,255), -1)
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255),3 )
                #cv2.imwrite(image_name, img)


            else:
                print("[*] {} skipped: {} {} {} {}.".format(image_name, xmin, ymin, xmax, ymax))



        # Write annotations.
        with open(os.path.join(dest_folder, 'annotations.json'), 'w+') as f:
            json.dump(output, f)

        # Create and write train and test indices.
        a = np.arange(len(output["annotations"]))
        np.random.shuffle(a)
        num_sample = a.shape[0]

        train_indices = a[:int(num_sample*0.9)]
        test_indices = a[int(num_sample*0.9):]

        np.save(os.path.join(dest_folder, "train-indices.npy"), train_indices)
        np.save(os.path.join(dest_folder, "test-indices.npy"), test_indices)

"""       
h2d = HUYFishEye2DATASET()    
print("HERE")
#h2d.run('/media/my_folder/DRONE_RACING_DATA/batch_2', 'DATASETS/shifted_batch_2')
#h2d.run('/media/my_folder/DRONE_RACING_DATA/batch_4', 'DATASETS/shifted_batch_4')
#h2d.run('/media/my_folder/DRONE_RACING_DATA/batch_5', 'DATASETS/shifted_batch_5')
#h2d.run('/media/my_folder/DRONE_RACING_DATA/batch_6', 'DATASETS/shifted_batch_6')
#h2d.run('/media/my_folder/DRONE_RACING_DATA/batch_7', 'DATASETS/shifted_batch_7')
"""
