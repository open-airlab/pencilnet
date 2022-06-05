import cv2
import numpy as np
import tensorflow as tf
import utils_io
import os

from directed_blur import KernelBuilder, DirectedBlurFilter
from pencil_filter import PencilFilter

class PencilNetInference:
    '''
    filters_to_apply - list of filters which will be applied to the frame before inference. The order is preserved.
    '''
    def __init__(self, h=160, w=120, filters_to_apply = []):
        self.shape = (h,w)
        self.filters_to_apply = filters_to_apply
        self.pencil_filter = PencilFilter()

    def initi_tf(self, model_path):
        print("[*] PencilNet: Model is being loaded from {}...".format(model_path))

        self.logger = Logger()
        self.logger.load(model_path)

        config = tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=2,
            allow_soft_placement=True
        )

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.session = tf.Session(config=config)
        checkpoints = self.logger.list_checkpoints()
        with self.session.as_default():
            self.model = self.logger.load_checkpoint(checkpoints[100]) # todo make as a param

        self.model._make_predict_function()
        print("[*] PencilNet: The model is loaded!")


    def run(self, bgr_img):
        # Preprocess data
        img = cv2.resize(bgr_img, (self.shape[0],self.shape[1])) # Resize image for the network.

        # Apply filters
        for pre_filter in self.filters_to_apply:
            img = pre_filter.apply(img)

        img = np.reshape(img, (1,120,160,3)) # 1 more channel is added to form a batch.
        img = img.astype(np.float32)/255. # Convert image type int8 to float32

        predictions = None
        # Run the network for inference!
        with self.session.as_default():
            with self.session.graph.as_default():
                predictions = self.model.predict(img)
        # -------------------------


        # -------------------------
        # Publish the cx, cy, distance, yaw_relative
        # -------------------------
        # Get results, pred_imd is for debugging, bbox is [(cx_on_img,cy_on_img,distance,yaw_relative)]
        pred_img, bboxes = utils_io.display_target_woWH(np.float32(predictions[0]), 
                                                            img[0], 
                                                            self.logger.config['output_shape'], 
                                                            self.threshold, ret=True)

        return pred_img, bboxes

# Test iference
if __name__ == '__main__':
    kernel_builder = KernelBuilder()
    # With kernel size you can control the simulated velocity of the drone => greater the size, the more the motion.
    kernel_size = 10
    kernel = kernel_builder.custom_rotated_horizontal_kernel(kernel_size, angle_in_degrees=0)
    directed_blur_filter = DirectedBlurFilter(kernel)

    test_img_path = os.path.join('Public_datasets', 'Test data', 'rgb_real_N_40', 'images','000735.png')
    test_img_bgr = cv2.imread(test_img_path)
    
    filters = [directed_blur_filter, PencilFilter]
    inference = PencilNetInference(filters_to_apply=filters)

    pred_img, bboxes = inference.run(test_img_bgr)

    center_x = [bbox[0] for bbox in bboxes]
    center_y = [bbox[1] for bbox in bboxes]
    distance = [bbox[2] for bbox in bboxes]
    yaw_relative = [bbox[3] for bbox in bboxes]

    cv2.imshow('Predicted', pred_img)