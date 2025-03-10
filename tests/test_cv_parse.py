import cv2
import unittest
import numpy as np
from app.cv_parse.SSDNet import SSDNet
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class CvTestCase(unittest.TestCase):
    def setUp(self):
        self.ssd_net = SSDNet(
            frozen_graph_path='models/frozen_inference_graph.pb', 
            pbtxt_path='models/go.pbtxt'
        )

    def tearDown(self):
        # 清理资源
        if hasattr(self.ssd_net, 'model'):
            if hasattr(self.ssd_net.model, 'sess'):
                self.ssd_net.model.sess.close()

    def test_model_loaded(self):
        self.assertIsNotNone(self.ssd_net.model)
        self.assertIsNotNone(self.ssd_net.__category_index__)

    def test_forward(self):
        image = cv2.imread('static/srcImage.jpg')
        if image is None:
            self.fail("测试图片加载失败")
            
        # 转换为RGB并创建batch
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_array = np.array([image, image, image, image])
        
        # 执行推理
        output = self.ssd_net.__forward__(input_array)  # Remove _SSDNet prefix
        
        # 验证输出
        self.assertIsNotNone(output)
        self.assertIn('num_detections', output)
        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)