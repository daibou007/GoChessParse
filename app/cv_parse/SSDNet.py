import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
import os
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

class SSDNet:

    def __init__(self, frozen_graph_path, pbtxt_path):
        '''
        初始化 SSD 网络检测器
        Args:
            frozen_graph_path: 训练好的模型权重文件路径（.pb文件）
            pbtxt_path: 标签映射文件路径（.pbtxt文件）
        '''
        try:
            self.model = self.__load_model__(frozen_graph_path)
            self.__category_index__ = self.__load_label_map__(pbtxt_path)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def __load_label_map__(self, pbtxt_path):
        '''加载标签映射文件
        Args:
            pbtxt_path: 标签映射文件路径（.pbtxt文件）
        Returns:
            category_index: 类别索引字典
        '''
        if not os.path.exists(pbtxt_path):
            raise FileNotFoundError(f"标签映射文件不存在: {pbtxt_path}")
            
        try:
            from object_detection.utils import label_map_util
            from object_detection.protos import string_int_label_map_pb2
            from google.protobuf import text_format
            
            with tf.io.gfile.GFile(pbtxt_path, 'r') as fid:
                label_map_string = fid.read()
                label_map = string_int_label_map_pb2.StringIntLabelMap()
                try:
                    text_format.Merge(label_map_string, label_map)
                except text_format.ParseError:
                    label_map.ParseFromString(label_map_string)
            
            return label_map_util.create_category_index(
                label_map_util.convert_label_map_to_categories(
                    label_map, max_num_classes=90, use_display_name=True))
                    
        except Exception as e:
            raise RuntimeError(f"标签映射文件加载失败: {str(e)}")


    def __load_model__(self, model_path):
        '''加载模型，支持 .pb 和 SavedModel 格式'''
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 尝试加载 SavedModel 格式
        if os.path.isdir(model_path):
            return tf.saved_model.load(model_path)
        
        # 加载 .pb 格式
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with open(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
            
            # 创建会话
            sess = tf.compat.v1.Session(graph=detection_graph)
            
            # 获取必要的张量
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)
            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            # 返回一个包装了会话和张量的对象
            return type('Model', (), {
                'sess': sess,
                'tensor_dict': tensor_dict,
                'image_tensor': image_tensor,
                '__call__': lambda self, x: sess.run(tensor_dict, feed_dict={image_tensor: x})
            })()

    def __old_load_label_map__(self, pbtxt_path):
        '''加载标签映射文件
        Args:
            pbtxt_path: 标签映射文件路径（.pbtxt文件）
        Returns:
            category_index: 类别索引字典
        '''
        if not os.path.exists(pbtxt_path):
            raise FileNotFoundError(f"标签映射文件不存在: {pbtxt_path}")
            
        try:
            from object_detection.utils import label_map_util
            return label_map_util.create_category_index_from_labelmap(
                pbtxt_path, use_display_name=True)
        except Exception as e:
            raise RuntimeError(f"标签映射文件加载失败: {str(e)}")

    def __forward__(self, image):
        '''执行推理，支持两种模型格式'''
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("输入必须是有效的numpy数组")
        
        if len(image.shape) != 4:
            raise ValueError("输入维度必须是4维 [batch_size, height, width, channels]")
        
        # 执行推理
        detections = self.model(image)
        
        # 统一返回格式
        return {
            'num_detections': detections['num_detections'] if isinstance(detections, dict) else detections[0],
            'detection_boxes': detections['detection_boxes'] if isinstance(detections, dict) else detections[1],
            'detection_scores': detections['detection_scores'] if isinstance(detections, dict) else detections[2],
            'detection_classes': detections['detection_classes'].astype(np.uint8) if isinstance(detections, dict) else detections[3].astype(np.uint8)
        }

    # ... 其他方法保持不变 ...

    def __split_image__(self, InputArray):
        '''
        :param InputArray:
        :return:
        '''
        roi_list = []
        srcImage = InputArray.copy()
        width, height = srcImage.shape[1], srcImage.shape[0]
        roi_width, roi_height = int(width * 0.6), int(height * 0.6)
        cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB, srcImage)
        for y in range(2):
            for x in range(2):
                roi_tlx, roi_tly = int(width * 0.4) * x, int(height * 0.4) * y
                ROI = srcImage[roi_tly:roi_tly + roi_height, roi_tlx:roi_tlx + roi_width]
                roi_list.append(ROI)
        roi_input = np.array([roi_list[0], roi_list[1], roi_list[2], roi_list[3]])
        return roi_input

    def __concat_image_output__(self, output_dict, image_shape, split_ROI_shape):
        center_list = []
        index = 0
        aver = 0
        width, height = image_shape[1], image_shape[0]
        for y in range(2):
            for x in range(2):
                roi_tlx, roi_tly = int(width * 0.4) * x, int(height * 0.4) * y
                divide_x, divide_y = width // 2, height // 2
                output_roi = {}
                output_roi['num_detections'] = int(output_dict['num_detections'][index])
                output_roi['detection_classes'] = output_dict[
                    'detection_classes'][index].astype(np.uint8)
                output_roi['detection_boxes'] = output_dict['detection_boxes'][index]
                output_roi['detection_scores'] = output_dict['detection_scores'][index]
                index += 1
                center_temp = self.__get_chess_pieces_position__(split_ROI_shape, output_roi, 0.1)
                for p in center_temp:
                    p = (int(p[0] + roi_tlx), int(p[1] + roi_tly), int(p[2]), (p[3][0] + roi_tlx, p[3][1] + roi_tly),
                         (p[4][0] + roi_tlx, p[4][1] + roi_tly))
                    if x == 0 and y == 0 and (p[0] > divide_x or p[1] > divide_y):
                        continue
                    elif x == 1 and y == 0 and (p[0] < divide_x or p[1] > divide_y):
                        continue
                    elif x == 0 and y == 1 and (p[0] > divide_x or p[1] < divide_y):
                        continue
                    elif x == 1 and y == 1 and (p[0] < divide_x or p[1] < divide_y):
                        continue
                    center_list.append(p)
                    aver += (abs(p[3][0] - p[4][0]) + abs(p[3][1] - p[4][1])) // 2
        if  len(center_list)==0:
            raise ZeroDivisionError
        aver /= len(center_list)
        for index, temp in enumerate(center_list):
            size_val = (abs(temp[3][0] - temp[4][0]) + abs(temp[3][1] - temp[4][1])) // 2
            if size_val < aver * 0.7:
                center_list.pop(index)
        return center_list

    def detect_chesspieces(self, InputArray):
        '''
        :param InputArray:
        :return:
        '''
        center_list = []
        roi_input = self.__split_image__(InputArray)
        output = self.__forward__(roi_input)
        center_list = self.__concat_image_output__(output, InputArray.shape, roi_input[0].shape)
        # print('size_val=', size_val)
        # self.draw_pts(InputArray.copy(), center_list)
        return center_list

    def __get_chess_pieces_position__(self, InputImage_shape, forward_result, beshowed_threshold=0.1):
        '''
        获取棋子位置
        '''
        h, w = InputImage_shape[0], InputImage_shape[1]
        chess_pieces_num = 0
        center_list = []
        for index, result in enumerate(forward_result['detection_scores']):
            if result < beshowed_threshold:
                break
            rect = forward_result['detection_boxes'][index]
            class_id = forward_result['detection_classes'][index]
            
            # 确保类别ID存在于category_index中
            if class_id not in self.__category_index__:
                continue
                
            try:
                pt1 = (int(rect[1] * w), int(rect[0] * h))
                pt2 = (int(rect[3] * w), int(rect[2] * h))
                center = (
                    (pt1[0] + pt2[0]) // 2, 
                    (pt1[1] + pt2[1]) // 2, 
                    class_id,  # 使用类别ID而不是类别名称
                    pt1, 
                    pt2
                )
                center_list.append(center)
                chess_pieces_num += 1
            except (TypeError, IndexError) as e:
                print(f"处理检测框时出错: {e}")
                continue
                
        return center_list

    def __old_get_chess_pieces_position__(self, InputImage_shape, forward_result, beshowed_threshold=0.1):
        '''

        :param InputImage_shape:
        :param forward_result:
        :param beshowed_threshold:
        :return:
        '''
        h, w = InputImage_shape[0], InputImage_shape[1]
        chess_pieces_num = 0
        center_list = []
        for index, result in enumerate(forward_result['detection_scores']):
            if result < beshowed_threshold:
                break
            rect = forward_result['detection_boxes'][index]
            class_name = self.__category_index__[forward_result['detection_classes'][index]]
            pt1 = ((int)(rect[1] * w), (int)(rect[0] * h))
            pt2 = ((int)(rect[3] * w), (int)(rect[2] * h))
            center = (
                (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2, forward_result['detection_classes'][index], pt1, pt2)
            center_list.append(center)
            chess_pieces_num += 1
        return center_list
