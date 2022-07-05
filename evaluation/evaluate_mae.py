import tensorflow as tf
# import tensorflow.compat.v1 as tf   # for this code to work with tensorflow 1.x
# tf.disable_v2_behavior()

import numpy as np
import cv2
import os

import datetime

from utils_io import SyntheticDataset

from logger import Logger
from metrics_iros import gate_center_mae_error, distance_mae_error, orientation_mae_error
from pencil_filter import PencilFilter
from other_filter import SobelEdgeExtractionFilter, CannyEdgeExtractionFilter

pencilFilter = PencilFilter()
sobelFilter = SobelEdgeExtractionFilter()
cannyFilter = CannyEdgeExtractionFilter()

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

class ModelEvaluate:

    def __init__(self, model_name, model_path, test_set_name, test_set_path, result_folder_path, filter_type="None"):
        
        self.model_name = model_name
        self.test_set_name = test_set_name

        self.result_folder_path = result_folder_path

        self.logger = Logger()
        self.logger.load(model_path) 
        self.config = self.logger.config

        self.config['dataset_folder'] = test_set_path
        # self.config['train_indices'] = os.path.join(test_set_path, 'train-indices.npy')
        self.config['test_indices'] = os.path.join(test_set_path, 'test-indices.npy')
    
        self.filter_type = filter_type


    def test_generator(self):
        """ Generator for training samples."""
        
        s = SyntheticDataset(self.config['dataset_folder'], grid_shape=(self.config['output_shape'][1], self.config['output_shape'][0]))

        indices = np.load(self.config['test_indices'])
        _, ref = s.get_data_by_index(indices[0])
        for i in indices:
            img, target = s.get_data_by_index(i)

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

            yield img/255., target



    def run(self):
        
            # Dataset preparation
            test_dataset = tf.data.Dataset.from_generator(generator = self.test_generator,
                                                        output_types = (tf.float32, tf.float32),
                                                        output_shapes=(tf.TensorShape(self.config['input_shape']), tf.TensorShape(self.config['output_shape'])))


            # Preprocess data.
            test_dataset = test_dataset.batch(32, drop_remainder=True)

            model = self.logger.load_checkpoint(epoch=100)

            model.compile(optimizer=self.logger.config['optimizer'],
                            loss=self.logger.loss,
                            metrics=[gate_center_mae_error, distance_mae_error, orientation_mae_error])
                            # metrics=[probability_gate_center_mae_error, probability_distance_mae_error, probability_orientation_mae_error])
            
            # Test the model.
            predictions = model.predict(test_dataset)
            result = model.evaluate(test_dataset)
            print(self.test_set_name, dict(zip(model.metrics_names, result)))

            # write to data
            # out_file = open( os.path.join(self.result_folder_path, self.model_name + ".txt"), "a+")
            out_file = open(os.path.join(self.result_folder_path, "MAE_" + now + ".txt"), "a+")
            out_file.write(self.model_name + " " + self.test_set_name+" "+str(dict(zip(model.metrics_names, result)))+"\n")
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
    test_data = ["sim_outdoor_combined", "rgb_real_N_100_from_New_Night", "rgb_real_N_40", "rgb_real_N_20", "rgb_real_N_10"]

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
            model_eval =  ModelEvaluate(model_name, model_path, test_set, test_set_path, result_folder_path, filter_type=filter_type)
            model_eval.run()

            tf.keras.backend.clear_session()

