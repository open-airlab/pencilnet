from cv2 import waitKey
import tensorflow as tf
# import tensorflow.compat.v1 as tf   # for this code to work with tensorflow 1.x
# tf.disable_v2_behavior()

import numpy as np
import cv2
import os
import itertools
import datetime
from utils_io import SyntheticDataset
from logger import Logger

from pencil_filter import PencilFilter
from other_filter import SobelEdgeExtractionFilter, CannyEdgeExtractionFilter

pencilFilter = PencilFilter()
sobelFilter = SobelEdgeExtractionFilter()
cannyFilter = CannyEdgeExtractionFilter()

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

class ModelEvaluate:

    def __init__(self, model_name, model_path, test_set_name, test_set_path, result_folder_path, filter_type="None", model_epoch=100):
        
        self.model_name = model_name
        self.test_set_name = test_set_name

        self.result_folder_path = result_folder_path
        # os.makedirs(self.result_folder_path)

        self.logger = Logger()
        self.logger.load(model_path) 
        self.config = self.logger.config

        self.config['dataset_folder'] = test_set_path
        # self.config['train_indices'] = os.path.join(test_set_path, 'train-indices.npy')
        self.config['test_indices'] = os.path.join(test_set_path, 'test-indices.npy')
    
        self.filter_type = filter_type

        self.threshold = 0.7

        self.model_epoch = model_epoch

        self.number_of_samples = 0
        self.false_negative = 0
        self.FN_percentage = 0


    def test_generator(self, index):
        """ Generator for training samples."""
        
        s = SyntheticDataset(self.config['dataset_folder'], grid_shape=(self.config['output_shape'][1], self.config['output_shape'][0]))


        # img, target = s.get_data_by_index(index)
        img, target = s.get_data_by_index_center_pixel_raw(index)

        # print("target = {}, with the size: {} ".format(target, len(target)))

        # apply pencil filter
        if self.filter_type == "pencil":
            img = pencilFilter.apply(img)
        elif self.filter_type == "sobel":
            img = sobelFilter.apply(img)
        elif self.filter_type == "canny":
            img = cannyFilter.apply(img)

        # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        # Normalize
        if img.shape != self.config['input_shape'] or target.shape!=self.config['output_shape']:
            img = cv2.resize(img, (self.config['input_shape'][1], self.config['input_shape'][0]), interpolation=cv2.INTER_NEAREST)

        return img/255., target


    def display_target_woWH(self, M, source_img, output_shape, threshold, ret=False):
        img = source_img.copy()
        img_height,img_width = img.shape[:2]

        bbox_results = []
        nrow, ncol = output_shape[0], output_shape[1]
        grid_dim_x = img_width/ncol
        grid_dim_y = img_height/nrow


        for i in range(3):
            for j in range(nrow):

                if M[i,j,0] > threshold:
                    cx, cy, distance, yaw_relative = M[i,j,1:]

                    cx_on_img = int(j*grid_dim_x + cx*grid_dim_x)
                    cy_on_img = int(i*grid_dim_y + cy*grid_dim_y)
                    cv2.circle(img, (cx_on_img, cy_on_img), 3, (0,0,1), 3)                

                    bbox_results.append((cx_on_img,cy_on_img,abs(float(distance)),yaw_relative))
                    
        #cv2.imwrite("display.png", img)
        if ret:
            return img, bbox_results


    def evaluate_false_negative(self, target, predictions, img, debug=False):

        pred_img, bboxes = self.display_target_woWH(np.float32(predictions[0]), 
                                                            img[0], 
                                                            self.logger.config['output_shape'], 
                                                            self.threshold, ret=True)

        self.number_of_samples += 1

        number_of_predictions = len(bboxes)

        if number_of_predictions < 1:
            self.false_negative += 1

        self.FN_percentage = 100 * self.false_negative / self.number_of_samples  # in percentage (%)


        if debug:
            print("Network made {} prediction(s), with metrics = {} and a rank_value = {}".format(len(bboxes), metrics, rank_value))
            cv2.imshow('Blured Images', pred_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return self.FN_percentage, self.false_negative, self.number_of_samples


    def run(self):
        
            # Load model

            model = self.logger.load_checkpoint(epoch=self.model_epoch)

            if self.filter_type == "pencil":
                model = self.logger.load_checkpoint(epoch=self.model_epoch)
                
            # model._make_predict_function()
            model.make_predict_function()

            indices = np.load(self.config['test_indices'])
            data_point = 0
            running_metrics = np.empty(3)
            for i in indices:
                img, target = self.test_generator(i)
                
                img = np.reshape(img, (1,120,160,3)) # 1 more channel is added to form a batch.


                # Test the model.
                predictions = model.predict(img)


                running_FN_percentage, _, number_of_samples = self.evaluate_false_negative(target, predictions, img)
                # running_FN_percentage, _, number_of_samples = self.evaluate_false_negative(target, predictions, img, debug=True)   # for seeing images


                data_point += 1
                print("Model {} Running ===> : {} %: FN percentage = {} %".format(self.model_name, data_point/len(indices)*100, running_FN_percentage))


            # write to data
            out_file = open( os.path.join(self.result_folder_path, "FN_" + now + ".txt"), "a+")
            out_file.write(self.model_name + " " + self.test_set_name+" FN%: "+ str(running_FN_percentage) +" num samples: "+ str(number_of_samples) +"\n")
            out_file.close()



if __name__ == '__main__':

    result_folder_path = "./evaluation_results"

    pencil_model_path = "/home/huy/transfer_learning_ws/src/pencil_filter_processor/pencil_perception_module/model/2022-01-22-13-45-pretrain-single-gate-corrected"
    sobel_model_path = "/home/huy/transfer_learning_ws/src/pencil_filter_processor/training/auto_labeling/trained_model/sobel-2022-06-06-17-30"
    canny_model_path = "/home/huy/transfer_learning_ws/src/pencil_filter_processor/training/auto_labeling/trained_model/canny-2022-06-07-17-35"
    GateNet_model_path = "/home/huy/transfer_learning_ws/src/pencil_filter_processor/Baselines/GateNet/test_folder/2022-01-24-01-06-rgb-on-sim"
    Adr_model_path = "/home/huy/transfer_learning_ws/src/pencil_filter_processor/Baselines/ADRNet/test_folder/ADR-mod-2022-02-16-16-03_adr_mod"
    DroNet_half_model_path = "/home/huy/transfer_learning_ws/src/pencil_filter_processor/Baselines/Dronet/test_folder/2022-02-15-01-12-dronet-half"
    DroNet_full_model_path = "/home/huy/transfer_learning_ws/src/pencil_filter_processor/Baselines/Dronet/test_folder/2022-02-15-13-29-dronet-full"
    

    base_test_data_folder_path = '/home/huy/dataset_ws/Test_data/RAL2022/rgb'
    # test_data = ["sim_outdoor_combined", "rgb_real_N_100_from_New_Night", "rgb_real_N_40", "rgb_real_N_20", "rgb_real_N_10"]
    test_data = ["rgb_real_N_10"]

    # base_test_data_folder_path = '/home/huy/dataset_ws/Test_data/RAL2022/blur'
    # test_data = ["real_40_2", "real_20_2", "real_10_2"]

    
    models = "GateNet sobel canny pencil".split()
    # models = "ADR".split()
    # models = "DroNet-Full".split()
    # models = "DroNet-Half".split()

    for model_name in models:

        if model_name == "pencil":
            model_path = pencil_model_path
            filter_type = "pencil"

        elif model_name == "sobel":
            model_path = sobel_model_path
            filter_type = "sobel"

        elif model_name == "canny":
            model_path = canny_model_path
            filter_type = "canny"

        elif model_name == "GateNet":
            model_path = GateNet_model_path
            filter_type = "None"

        elif model_name == "ADR":
            model_path = Adr_model_path
            filter_type = "None"

        elif model_name == "DroNet-Half":
            model_path = DroNet_half_model_path
            filter_type = "None"

        elif model_name == "DroNet-Full":
            model_path = DroNet_full_model_path
            filter_type = "None"

        else:
            print("Error, no model is selected.")


        for test_set in test_data:
            print("Model name: {}, with filter type: {}".format(model_name, filter_type))
            print("test folder: {}".format(test_set))
            
            test_set_path = os.path.join(base_test_data_folder_path, test_set)


            model_eval =  ModelEvaluate(model_name, model_path, test_set, test_set_path, result_folder_path, filter_type=filter_type, model_epoch=100)
            model_eval.run()

