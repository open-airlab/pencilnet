import json
import os
import cv2
import numpy as np
from shutil import copyfile
import math

class SyntheticDataset:
    def __init__(self, path_to_dataset, grid_shape=(5, 5)):
        self._path_to_dataset = path_to_dataset
        json_file = open(os.path.join(self._path_to_dataset, 'annotations.json'))
        data = json.load(json_file)
        self.annotations = data['annotations']

        self.num_samples = len(self.annotations)

        self.num_grid_x, self.num_grid_y = grid_shape

    def display_target(self, M, source_img, ret=False):
        img = source_img.copy()
        img_height,img_width = img.shape[:2]
        grid_dim_x = img_width/self.num_grid_x
        grid_dim_y = img_height/self.num_grid_y

        annotations = []
        for i in range(self.num_grid_y):
            for j in range(self.num_grid_x):
                #cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
                #cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

                if M[i,j,0] > 0.8:
                    cx, cy,  distance, yaw_relative = M[i,j,1:]
                    rw, rh = 0, 0
                    print(M[i,j,0],i,j,cx,cy,rh, distance, yaw_relative)

                    cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                    cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                    cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,255), 3)                

                    h = int(rh*img_height)
                    w = int(rw*img_width)

                    cv2.rectangle(img, (int(cx_on_img-(w/2)), int(cy_on_img-(h/2))), 
                                        (int(cx_on_img+(w/2)), int(cy_on_img+(h/2))), (0,0,255), 3)

                    # Unnormalize
                    #distance = distance*20.
                    #yaw_relative = (yaw_relative*np.pi - np.pi)                   

                    yaw_relative = yaw_relative/np.pi*180.

                    rel_text = "{:3.2f}".format(abs(distance))+" m."
                    yaw_text = "{:3.2f}".format(yaw_relative)+" deg." 

                    cv2.putText(img, rel_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, lineType=cv2.LINE_AA)                      
                    cv2.putText(img, yaw_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, lineType=cv2.LINE_AA)                      

                    annotations.append([cx_on_img, cy_on_img, abs(float(distance))])

        #cv2.imwrite("display.png", img)
        if ret:
            return img, annotations



    def display_target_large_annot(self, M, source_img, ret=False):
        img = source_img.copy()
        img_height,img_width = img.shape[:2]
        grid_dim_x = img_width/self.num_grid_x
        grid_dim_y = img_height/self.num_grid_y

        annotations = []
        for i in range(self.num_grid_y):
            for j in range(self.num_grid_x):
                #cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
                #cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

                if M[i,j,0] > 0.8:
                    cx, cy,  gx, gy, gz, yaw_relative = M[i,j,1:]
                    rw, rh = 0, 0
                    print(M[i,j,0],i,j,cx,cy,gx,gy,gz, yaw_relative)

                    cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                    cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                    cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,255), 3)                

                    h = int(rh*img_height)
                    w = int(rw*img_width)

                    cv2.rectangle(img, (int(cx_on_img-(w/2)), int(cy_on_img-(h/2))), 
                                        (int(cx_on_img+(w/2)), int(cy_on_img+(h/2))), (0,0,255), 3)

                    # Unnormalize
                    #distance = distance*20.
                    #yaw_relative = (yaw_relative*np.pi - np.pi)                   

                    yaw_relative = yaw_relative/np.pi*180.

                    distance = np.sqrt(gx**2 + gy**2 + gz**2)
                    rel_text = "{:3.2f}".format(abs(distance))+" m."
                    yaw_text = "{:3.2f}".format(yaw_relative)+" deg." 

                    cv2.putText(img, rel_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, lineType=cv2.LINE_AA)                      
                    cv2.putText(img, yaw_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, lineType=cv2.LINE_AA)                      

                    annotations.append([cx_on_img, cy_on_img, abs(float(distance))])

        #cv2.imwrite("display.png", img)
        if ret:
            return img, annotations




    def get_data_by_index(self, index, save_img=False, show_grid=False):

        annotation = self.annotations[index]

        img = cv2.imread(os.path.join(self._path_to_dataset, 'images', annotation['image']))
        ori = img.copy()
        assert(img is not None)

        img_height,img_width = img.shape[:2]
        grid_dim_x = img_width/self.num_grid_x
        grid_dim_y = img_height/self.num_grid_y
        
        # prob,cx,cy,rw,rh,distance, yaw_rel
        M = np.zeros( (self.num_grid_y, self.num_grid_x, 5), dtype=np.float32)
        for annot in annotation['annotations']:

            # Read bbox.
            x = annot['xmin']
            y = annot['ymin']
            w = annot['xmax']-annot['xmin']
            h = annot['ymax']-annot['ymin']

            if "center_x" in annot.keys():
                cx, cy = annot["center_x"], annot["center_y"]
            
            else:  
                x = annot['xmin']
                y = annot['ymin']
                w = annot['xmax']-annot['xmin']
                h = annot['ymax']-annot['ymin']
                cx, cy = x+(w/2.), y+(h/2.)
                assert(True)
                assert(False)
                exit()
                return None

            x_index = int(cx/grid_dim_x)
            y_index = int(cy/grid_dim_y)
            #print(x_index, y_index)

            cx, cy = (cx % grid_dim_x)/grid_dim_x, (cy % grid_dim_y)/grid_dim_y
            rw, rh = w/img_width, h/img_height

            # Read distance and orientation.
            distance = annot['distance']

            """
            if annot['yaw_relative']>0:
                yaw_relative=1
            else:    
                yaw_relative=-1
            """
            yaw_relative = annot['yaw_relative']



            #yaw_relative = math.degrees(yaw_relative)
            #if yaw_relative<0:
            #    yaw_relative = (2*np.pi)+yaw_relative

            #if yaw_relative<0 or yaw_relative>2*np.pi:
            #    assert("Yaw relative {}".format(yaw_relative))    
            # Normalize
            #distance = distance/20
            #yaw_relative = (yaw_relative+np.pi)/(np.pi)
            #if yaw_relative > 2 or yaw_relative<0:
            #    print(yaw_relative)
            #    assert yaw_relative>1
            #    exit()

            #if distance > 1:
            #    print(distance)
            #    assert distance>1
            #    exit()

            M[y_index, x_index]=np.array([1., cx, cy, distance, yaw_relative])

        if save_img:
            for annot in annotation['annotations']:
                cv2.rectangle(img, (annot['xmin'], annot['ymin']), (annot['xmax'], annot['ymax']), (0,0,255), 3)

            if show_grid:
                for i in range(self.num_grid_y+1):
                    for j in range(self.num_grid_x+1):
                        cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
                        cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

            cv2.imwrite('img.png', img)
            self.display_target(M, ori)

        return img, M


    def get_data_by_index_center_pixel_raw(self, index, save_img=False, show_grid=False):
        '''
        This function returns label sample with raw pixel value of cx and cy, rather than relative and normalized values.
        Intended for EVALUATION, NOT FOR TRAINING
        The Target (M) returned is DIFFERENT with the training target!!!
        '''

        annotation = self.annotations[index]

        img = cv2.imread(os.path.join(self._path_to_dataset, 'images', annotation['image']))
        ori = img.copy()
        assert(img is not None)

        img_height,img_width = img.shape[:2]
        grid_dim_x = img_width/self.num_grid_x
        grid_dim_y = img_height/self.num_grid_y
        

        evaluate_target = []

        for annot in annotation['annotations']:

            cx, cy = annot["center_x"], annot["center_y"]
            
            x_index = int(cx/grid_dim_x)
            y_index = int(cy/grid_dim_y)

            # Read distance and orientation.
            distance = annot['distance']
            yaw_relative = annot['yaw_relative']

            evaluate_target.append(np.array([ int(cx), int(cy), float(distance), float(yaw_relative)]))

        return img, evaluate_target   # image, target





    def get_data_by_index_second_version(self, index, save_img=False, show_grid=False):

        annotation = self.annotations[index]

        img = cv2.imread(os.path.join(self._path_to_dataset, 'images', annotation['image']))
        ori = img.copy()
        assert(img is not None)

        img_height,img_width = img.shape[:2]
        grid_dim_x = img_width/self.num_grid_x
        grid_dim_y = img_height/self.num_grid_y
        
        # prob,cx,cy,rw,rh,distance, yaw_rel
        M = np.zeros( (self.num_grid_y, self.num_grid_x, 7), dtype=np.float32)
        for annot in annotation['annotations']:

            # Read bbox.
            x = annot['xmin']
            y = annot['ymin']
            w = annot['xmax']-annot['xmin']
            h = annot['ymax']-annot['ymin']

            if "center_x" in annot.keys():
                cx, cy = annot["center_x"], annot["center_y"]
            
            else:  
                x = annot['xmin']
                y = annot['ymin']
                w = annot['xmax']-annot['xmin']
                h = annot['ymax']-annot['ymin']
                cx, cy = x+(w/2.), y+(h/2.)
                assert(True)
                assert(False)
                exit()
                return None

            x_index = int(cx/grid_dim_x)
            y_index = int(cy/grid_dim_y)

            gx = annot["gx"]
            gy = annot["gy"]
            gz = annot["gz"]

            cx, cy = (cx % grid_dim_x)/grid_dim_x, (cy % grid_dim_y)/grid_dim_y
            rw, rh = w/img_width, h/img_height

            # Read distance and orientation.
            distance = annot['distance']

            """
            if annot['yaw_relative']>0:
                yaw_relative=1
            else:    
                yaw_relative=-1
            """
            yaw_relative = annot['yaw_relative']



            #yaw_relative = math.degrees(yaw_relative)
            #if yaw_relative<0:
            #    yaw_relative = (2*np.pi)+yaw_relative

            #if yaw_relative<0 or yaw_relative>2*np.pi:
            #    assert("Yaw relative {}".format(yaw_relative))    
            # Normalize
            #distance = distance/20
            #yaw_relative = (yaw_relative+np.pi)/(np.pi)
            #if yaw_relative > 2 or yaw_relative<0:
            #    print(yaw_relative)
            #    assert yaw_relative>1
            #    exit()

            #if distance > 1:
            #    print(distance)
            #    assert distance>1
            #    exit()

            M[y_index, x_index]=np.array([1., cx, cy, gx, gy, gz, yaw_relative])

        if save_img:
            for annot in annotation['annotations']:
                cv2.rectangle(img, (annot['xmin'], annot['ymin']), (annot['xmax'], annot['ymax']), (0,0,255), 3)

            if show_grid:
                for i in range(self.num_grid_y+1):
                    for j in range(self.num_grid_x+1):
                        cv2.line(img, (int(j*grid_dim_x), 0), (int(j*grid_dim_x), img_height), (0,255,0), 1)
                        cv2.line(img, (0, int(i*grid_dim_y)), (img_width, int(i*grid_dim_y)), (0,255,0), 1)

            cv2.imwrite('img.png', img)
            self.display_target(M, ori)

        return img, M







class MATLABoutput2json:
    def __init__(self):
        pass

    def run(self, matlab_output_fname, image_folder, dest_folder):
        matlab_file = open(matlab_output_fname, 'r')

        if not os.path.isdir(dest_folder):
            print("%s created." % dest_folder)
            os.mkdir(dest_folder)
            os.mkdir(os.path.join(dest_folder, 'images'))


        output = {"annotations":[]}
        for line in matlab_file.readlines():
            code, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, o, d = line.split(' ')
            p1x = float(p1x)
            p2x = float(p2x)
            p3x = float(p3x)
            p4x = float(p4x)
            p1y = float(p1y)
            p2y = float(p2y)
            p3y = float(p3y)
            p4y = float(p4y)
            d = float(d)
            o = float(o[:-1])

            image_name = code+'.png'
            #print(image_name)
            img = cv2.imread(image_folder+'/'+image_name)
            #cv2.circle(img,(int(float(p1x)),int(float(p1y))),3,(0,0,255))
            #cv2.circle(img,(int(float(p2x)),int(float(p2y))),3,(0,0,255))
            #cv2.circle(img,(int(float(p3x)),int(float(p3y))),3,(0,0,255))
            #cv2.circle(img,(int(float(p4x)),int(float(p4y))),3,(0,0,255))

            xmin = int(min(p1x,p2x,p3x,p4x))
            ymin = int(min(p1y,p2y,p3y,p4y))
            xmax = int(max(p1x,p2x,p3x,p4x))
            ymax = int(max(p1y,p2y,p3y,p4y))
            #cv2.rectangle(img, (xmin, ymin), (xmax,ymax), (0,0,255), 3)
            #cv2.imwrite('temp.png', img)
        
            if xmin>0 and ymin>0 and xmax<img.shape[1] and ymax<img.shape[0]:
                sample = {"image":image_name, "annotations":[{"class_id":1, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "distance":d, "rotation":o}]}
                output['annotations'].append(sample)
                copyfile(os.path.join(image_folder, image_name), os.path.join(dest_folder,'images', image_name))
                
            """    
            else:
                print(image_name, xmin, ymin, xmax, ymax, d, o)
                cv2.circle(img,(int(float(p1x)),int(float(p1y))),3,(0,0,255))
                cv2.circle(img,(int(float(p2x)),int(float(p2y))),3,(0,0,255))
                cv2.circle(img,(int(float(p3x)),int(float(p3y))),3,(0,0,255))
                cv2.circle(img,(int(float(p4x)),int(float(p4y))),3,(0,0,255))
                cv2.rectangle(img, (xmin, ymin), (xmax,ymax), (0,0,255), 3)
                cv2.imwrite(image_name, img)
            """

        with open(os.path.join(dest_folder, 'annotations.json'), 'w+') as f:
            json.dump(output, f)

       
        a = np.arange(len(output["annotations"]))
        np.random.shuffle(a)
        num_sample = a.shape[0]
        print(a)
        print(a.shape)

        train_indices = a[:int(num_sample*0.9)]
        test_indices = a[int(num_sample*0.9):]

        print(train_indices)
        print(test_indices)
        np.save(os.path.join(dest_folder, "train-indices.npy"), train_indices)
        np.save(os.path.join(dest_folder, "test-indices.npy"), test_indices)


#instance = MATLABoutput2json()
#instance.run('new_data3.txt', '/home/ilkerbozcan/Downloads/single_gate/new_data3' ,'andriy_data')        
