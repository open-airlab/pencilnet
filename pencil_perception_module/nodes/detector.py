#!/usr/bin/env python
from __future__ import print_function

from pencil_filter import PencilFilter

import rospy
import cv2
from std_msgs.msg import String, Float64, Int16
from pencil_perception_module.msg import PredictionStamped
from geometry_msgs.msg import PolygonStamped, Point32, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import numpy as np
import utils_io
import time
import sys

from logger import Logger

class Detector:

  def __init__(self):

    # Init node.
    rospy.init_node('detector', anonymous=True)

    # Load model.
    model_path = rospy.get_param("~model_path")
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
    with self.session.as_default():
      self.model = self.logger.load_checkpoint(rospy.get_param("~epoch"))

      self.model._make_predict_function()
      print("Epoch {} ({})".format(rospy.get_param("~epoch"), type(rospy.get_param("~epoch"))))
    print("[*] PencilNet: The model is loaded!")


    # Get image display threshold.
    self.threshold = rospy.get_param("~threshold")
    print("[*] PencilNet: Display detections with threshold {}.".format(self.threshold))

    # Subscribe image topic.
    subscribed_topic_name = rospy.get_param("~subscribed_topic_name")
    self.image_sub = rospy.Subscriber(subscribed_topic_name,Image,self.callback)
    print("[*] PencilNet: Detector is subscribed to {}".format(subscribed_topic_name))

    self.gt_center_sub = rospy.Subscriber("/real_center",Point,self.update_gt_center)
    self.gt_gate_center_in_pixels = (-1, -1)


    # To be used to convert ros image to cv2 images.
    self.bridge = CvBridge()

    # Image publisher for inspection of gate center prediction.
    self.img_pub = rospy.Publisher("/predicted_center_on_image", Image, queue_size=10)

    published_topic_name = rospy.get_param("~published_topic_name")
    self.pub = rospy.Publisher(published_topic_name, PredictionStamped, queue_size=10)
    # self.pub_real_distance = rospy.Publisher("/real_distance_ref", Float64, queue_size=10)
    self.pub_predicted_distance = rospy.Publisher("/predicted_distance", Float64, queue_size=10)
    self.pub_predicted_angle = rospy.Publisher("/predicted_angle", Float64, queue_size=10)
    self.pub_predicted_2d = rospy.Publisher("/gates/detected_2D", PolygonStamped, queue_size=10)
    print("[*] PencilNet: Detector results are published as topic {}".format(published_topic_name))

    self.window = []
    self.windows_size = 1
    
    self.apply_pencil_filter = rospy.get_param("~apply_pencil_filter")
    if self.apply_pencil_filter:
      self.pencil_filter = PencilFilter()
      print("[*] PencilNet: Pencil filter is applied.")
    else:
      print("[*] PencilNet: No pencil filter is applied.")

  def update_gt_center(self, point_msg):
    self.gt_gate_center_in_pixels = (int(point_msg.x), int(point_msg.y))


  def callback(self,img_msg):
    current_time = rospy.Time.now()
    image_time = rospy.Time()
    image_time = img_msg.header.stamp

    if ((current_time.to_sec() - image_time.to_sec()) <= 0.025):  #only process image up to 0.1 sec late

      try:
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
      except CvBridgeError as e:
        print(e)

      # -------------------------
      # Detection starts here
      # -------------------------

      # Preprocess data
      start_time = time.time()
      h,w,d = self.logger.config['input_shape']
      img = cv2.resize(cv_image, (160,120)) # Resize image for the network.
      
      # Apply Pencil filter
      pencil_start_time = time.time()
      if self.apply_pencil_filter:
        img = self.pencil_filter.apply(img)
      pencil_end_time = time.time()
      
      img = np.reshape(img, (1,120,160,3)) # 1 more channel is added to form a batch.
      img = img.astype(np.float32)/255. # Convert image type int8 to float32

      # Run the network for inference!
      start_time = time.time()
      with self.session.as_default():
        with self.session.graph.as_default():
          predictions = self.model.predict(img)
      end_time = time.time()
      print("[*] PencilNet: The network prediction takes a total of {} Millisecs.".format((end_time-start_time)*1000) )
      print("[*] PencilNet: Pencil filter takes {} Millisecs.".format((pencil_end_time - pencil_start_time)*1000) )

      # -------------------------


      # -------------------------
      # Publish the cx, cy, distance, yaw_relative
      # -------------------------
      # Get results, pred_imd is for debugging, bbox is [(cx_on_img,cy_on_img,distance,yaw_relative)]
      pred_img, bboxes = utils_io.display_target_woWH(np.float32(predictions[0]), 
                                                          img[0], 
                                                          self.logger.config['output_shape'], 
                                                          self.threshold, ret=True)


      pred_imgmsg = self.bridge.cv2_to_imgmsg(np.uint8(pred_img*255), "bgr8")
      self.img_pub.publish(pred_imgmsg)


      center_x = [Int16(bbox[0]) for bbox in bboxes]
      center_y = [Int16(bbox[1]) for bbox in bboxes]
      distance = [Float64(bbox[2]) for bbox in bboxes]
      yaw_relative = [Float64(bbox[3]) for bbox in bboxes]

      msg = PredictionStamped()
      msg.header.stamp = rospy.Time.now()
      msg.header.frame_id = "image_frame"
      msg.predicted_center_x = center_x
      msg.predicted_center_y = center_y
      msg.predicted_distance = distance
      msg.predicted_yaw = yaw_relative
      self.pub.publish(msg)

    else:
      print (current_time.to_sec() - image_time.to_sec())
      rospy.logwarn("Delayed message detected. Discard the frame.")


def main(args):
  print("[*] PencilNet: Detector node is started.")
  detector = Detector()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  detector.out.release()  
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
