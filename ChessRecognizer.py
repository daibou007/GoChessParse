import cv2
import numpy as np
import os
from app.cv_parse.SSDNet import SSDNet
from app.cv_parse.ChessBoardParser import ChessBoardParser

class ChessRecognizer:
    def __init__(self, model_path='models/frozen_inference_graph.pb', label_path='models/go.pbtxt'):
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 设置默认模型路径
        if model_path is None:
            model_path = os.path.join(project_root, 'models', 'frozen_inference_graph.pb')
        if label_path is None:
            label_path = os.path.join(project_root, 'models', 'go.pbtxt')
            
        # 验证文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件不存在: {label_path}")
            
        self.ssd_net = SSDNet(frozen_graph_path=model_path, pbtxt_path=label_path)
        self.parser = ChessBoardParser()

    def recognize_from_file(self, image_path, show_result=False, save_result=False):
        """从图片文件识别棋盘"""
        print(f"\n开始处理图片: {image_path}")
        
        # 读取并预处理图片
        print("正在读取图片...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        print(f"调整图片尺寸为 600x600...")
        image = cv2.resize(image, (600, 600))
        
        # 检测棋子
        print("正在检测棋子位置...")
        try:
            center_lists = self.ssd_net.detect_chesspieces(InputArray=image)
            print(f"检测到 {len(center_lists)} 个棋子")
        except ZeroDivisionError:
            print("错误: 未检测到有效的棋子")
            raise RuntimeError("未检测到有效的棋子")
        
        # 解析棋盘
        print("正在解析棋盘布局...")
        output_matrix = self.parser.output(image, center_lists)
        print(f"棋盘大小: {output_matrix.shape}")
        
        # 处理显示和保存
        result_image = None
        if show_result or save_result:
            print("正在生成结果图像...")
            result_image = ChessBoardParser.draw_chesspieces_locate(
                image=image.copy(), 
                center_lists=center_lists
            )
            
            if show_result:
                print("显示识别结果...")
                cv2.imshow('识别结果', result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if save_result:
                output_path = image_path.rsplit('.', 1)[0] + '_result.jpg'
                print(f"保存结果图像到: {output_path}")
                cv2.imwrite(output_path, result_image)
        
        print("处理完成\n")
        return output_matrix, result_image
