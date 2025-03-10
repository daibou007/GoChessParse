import cv2
import numpy as np

class ChessBoardParser:
    def __init__(self):
        pass

    def output(self, srcImage, center_list):
        print("\n开始处理棋盘图像...")
        print(f"输入图像尺寸: {srcImage.shape}")
        print(f"检测到棋子数量: {len(center_list)}")
        
        print("创建棋子掩码...")
        maskImage = np.zeros((srcImage.shape[0], srcImage.shape[1]), np.uint8)
        
        print("重建棋盘边缘...")
        edgeImage = self.__rebuildChessboard__(srcImage, center_list)
        
        print("检测棋盘边界线...")
        edge_lines = self.__houghEdge__(edgeImage, srcImage)
        
        corners_list = []
        print("生成棋子掩码...")
        self.__get__pieces_mask(maskImage, center_list)
        
        print("计算边界交点...")
        for index in range(len(edge_lines)):
            corners_list.append(self.__clac_intersection(edge_lines[index], edge_lines[(index + 1) % 4]))
        print(f"找到 {len(corners_list)} 个边界交点")
        
        print("透视变换校正...")
        dstImage, center_list = self.__remapLocate__(edgeImage, corners_list, maskImage, srcImage.shape[0:2])
        
        print("分析棋盘网格...")
        chess_board_pos = self.__shadowHist__(dstImage)
        chess_board_pos = self.__validate_edge_line__(chess_board_pos, dstImage, center_list)
        
        print("生成棋盘矩阵...")
        output_matrix = self.__position__(chess_board_pos, center_list)
        print(f"棋盘矩阵尺寸: {output_matrix.shape}")
        print("处理完成\n")
        
        return output_matrix

    @staticmethod
    def draw_chesspieces_locate(image, center_lists):
        """绘制棋子位置
        Args:
            image: 输入图像
            center_lists: 棋子位置列表
        Returns:
            标注后的图像
        """
        result_image = image.copy()
        for center in center_lists:
            # 确保所有坐标都是整数
            x, y = int(center[0]), int(center[1])
            class_id = int(center[2])
            pt1 = tuple(map(int, center[3]))
            pt2 = tuple(map(int, center[4]))
            
            # 绘制检测框
            cv2.rectangle(result_image, pt1, pt2, (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
            
            # 添加类别标签
            label = f"Class {class_id}"
            cv2.putText(result_image, label, (x-10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                       
        return result_image

    def __get__pieces_mask(self,maskImage,center_list):
        for center in center_list:
            color = 70
            if center[2] == 2:
                color = 255
            # 确保坐标为整数类型
            center_x = int(center[0])
            center_y = int(center[1])
            # 修正circle参数:第二个参数需要是tuple类型的center坐标
            cv2.circle(maskImage, (center_x, center_y), 8, (color), -1)

    def __clac_intersection(self, line_a, line_b):
        x1_a, y1_a, x2_a, y2_a = line_a
        x1_b, y1_b, x2_b, y2_b = line_b
        A_a = y2_a - y1_a
        B_a = x1_a - x2_a
        C_a = x2_a * y1_a - x1_a * y2_a
        A_b = y2_b - y1_b
        B_b = x1_b - x2_b
        C_b = x2_b * y1_b - x1_b * y2_b
        m = A_a * B_b - A_b * B_a
        output_x = (C_b * B_a - C_a * B_b) / m
        output_y = (C_a * A_b - C_b * A_a) / m
        return (int(output_x), int(output_y))

    def __detect__(self, srcImage):
        """检测棋子位置
        Args:
            srcImage: 输入图像
        Returns:
            center_list: 棋子位置列表
        """
        # 初始化检测网络
        if not hasattr(self, '_ChessBoardParser__net'):
            self.__net = ChessPieceDetector()
            
        # 执行棋子检测
        center_list = self.__net.chess_piece_mark(srcImage)
        return center_list

    def __rebuildChessboard__(self, srcImage, center_list, padding_val=8):
        InputArray = srcImage.copy()
        cv2.GaussianBlur(InputArray, (3, 3), 1, InputArray)
        edgeImage = cv2.Canny(InputArray, 20, 80)
        for pt in center_list:
            # clear bg
            cv2.rectangle(edgeImage, (pt[3][0] - padding_val, pt[3][1] - padding_val),
                          (pt[4][0] + padding_val, pt[4][1] + padding_val), (0, 0, 0), -1)
            cv2.line(edgeImage, ((pt[3][0] + pt[4][0]) // 2, pt[3][1] - padding_val),
                     ((pt[3][0] + pt[4][0]) // 2, pt[4][1] + padding_val),
                     (255, 255, 255), 2)
            cv2.line(edgeImage, (pt[3][0] - padding_val, (pt[3][1] + pt[4][1]) // 2),
                     (pt[4][0] + padding_val, (pt[3][1] + pt[4][1]) // 2),
                     (255, 255, 255), 2)
        return edgeImage

    def __validate_edge_line__(self, chess_board_pos, edgeImage, center_list):
        print("\n开始验证边界线...")
        temp_way_num = min(len(chess_board_pos[0]), len(chess_board_pos[1]))
        print(f"当前网格尺寸: {len(chess_board_pos[0])}x{len(chess_board_pos[1])}")
        
        perhaps_ways = [9, 13, 19]
        delta_ways = []
        print("执行形态学处理...")
        edgeImage = cv2.dilate(edgeImage, cv2.getStructuringElement(0, (12, 12)))
        _, edgeImage = cv2.threshold(edgeImage, 125, 255, cv2.THRESH_BINARY_INV)
        
        print("分析轮廓信息...")
        contours, hier = cv2.findContours(edgeImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        print(f"检测到 {len(contours)} 个轮廓")
        
        con_inf = []
        for con in contours:
            x, y, w, h = cv2.boundingRect(con)
            con_center = (x + w // 2, y + h // 2)
            area = abs(cv2.contourArea(con))
            con_inf.append((con_center, area))
            
        print("计算最佳网格大小...")
        for way in perhaps_ways:
            delta_ways.append(abs(way - temp_way_num))
        way_num = perhaps_ways[delta_ways.index(min(delta_ways))]
        print(f"选择网格大小: {way_num}")
        
        print("生成候选区域...")
        x_pairs, y_pairs = [], []
        parser_image = np.zeros([600, 600, 3], np.uint8)
        parser_image[:, :, 0] = edgeImage.copy() * 0.85
        
        for pos_index in range(len(chess_board_pos[0]) - way_num + 1):
            pos_list = chess_board_pos[0][pos_index:pos_index + way_num]
            y_pairs.append([pos_list[0], pos_list[-1]])
        for pos_index in range(len(chess_board_pos[1]) - way_num + 1):
            pos_list = chess_board_pos[1][pos_index:pos_index + way_num]
            x_pairs.append([pos_list[0], pos_list[-1]])
            
        print(f"生成 {len(x_pairs)}x{len(y_pairs)} 个候选区域")
        
        roi_list = []
        for y_pair in y_pairs:
            for x_pair in x_pairs:
                pt1 = (x_pair[0], y_pair[0])
                pt2 = (x_pair[1], y_pair[1])
                roi_list.append((pt1, pt2))
                
        print("分析候选区域...")
        roi_contour_anyis = []
        for i, roi in enumerate(roi_list):
            pt1, pt2 = roi[0], roi[1]
            con_num = 0
            for inf in con_inf:
                center = inf[0]
                if center[0] > pt1[0] and center[0] < pt2[0] and center[1] > pt1[1] and center[1] < pt2[1]:
                    con_num += 1
                    cv2.circle(parser_image, center, 3, (0, 255, 0), -1)
            cv2.rectangle(parser_image, pt1, pt2, (0, 255, 0), 1)
            parser_image = np.zeros([600, 600, 3], np.uint8)
            parser_image[:, :, 0] = edgeImage.copy()
            print(f"区域 {i+1}/{len(roi_list)} 包含 {con_num} 个轮廓")
            roi_contour_anyis.append(con_num)
            
        if len(roi_contour_anyis) <= 0:
            print("未找到有效区域，保持原始网格")
            return chess_board_pos
            
        print("选择最佳区域...")
        roi_index = roi_contour_anyis.index(max(roi_contour_anyis))
        output_roi = roi_list[roi_index]
        print(f"边界范围:")
        print(f"x: {output_roi[0][0]} -> {output_roi[1][0]}")
        print(f"y: {output_roi[0][1]} -> {output_roi[1][1]}")
        
        print("过滤网格点...")
        chess_board_pos[1] = list(filter(lambda pos: True if pos >= output_roi[0][0] else False, chess_board_pos[1]))
        chess_board_pos[1] = list(filter(lambda pos: True if pos <= output_roi[1][0] else False, chess_board_pos[1]))
        chess_board_pos[0] = list(filter(lambda pos: True if pos >= output_roi[0][1] else False, chess_board_pos[0]))
        chess_board_pos[0] = list(filter(lambda pos: True if pos <= output_roi[1][1] else False, chess_board_pos[0]))
        
        print(f"最终网格尺寸: {len(chess_board_pos[0])}x{len(chess_board_pos[1])}\n")
        return chess_board_pos

    def __houghEdge__(self, edgeImage, srcImage=None):
        thresh_min = min(edgeImage.shape)
        # 修正HoughLinesP参数，rho必须为整数
        lines = cv2.HoughLinesP(edgeImage, 
                               rho=1,  # rho参数应为整数
                               theta=np.pi/180, 
                               threshold=160,
                               minLineLength=int(edgeImage.shape[0] * 0.7),
                               maxLineGap=int(thresh_min * 0.5))
        
        # 默认返回值修改为正确的格式
        default_lines = [
            [0, 1, 5, 21],  # x1, y1, x2, y2
            [8, 1, 5, 3],
            [4, 12, 0, 8],
            [41, 11, 20, 15]
        ]
        
        if lines is None:
            return default_lines
            
        # 转换线段格式
        lines = [line[0] for line in lines]  # 展平数组
        
        # 筛选水平和垂直线
        lines_h = [line for line in lines if abs(line[1] - line[3]) > edgeImage.shape[0] * 0.5]
        lines_v = [line for line in lines if abs(line[0] - line[2]) > edgeImage.shape[1] * 0.5]
        
        # 排序
        lines_h = sorted(lines_h, key=lambda x: x[0])
        lines_v = sorted(lines_v, key=lambda x: x[1])
        
        # 调试绘制（如果提供了源图像）
        if srcImage is not None:
            for line in lines:
                pt1, pt2 = (int(line[0]), int(line[1])), (int(line[2]), int(line[3]))
                cv2.line(srcImage, pt1, pt2, (0, 0, 255), 2)
        
        # 返回四条边界线
        if len(lines_h) < 2 or len(lines_v) < 2:
            return default_lines
            
        return [
            lines_h[0],      # 上边界
            lines_v[0],      # 左边界
            lines_h[-1],     # 下边界
            lines_v[-1]      # 右边界
        ]


    def __remapLocate__(self, edgeImage, corner_list, maskImage, output_Imageshape=(600, 600)):
        prespect_mat, dstImage = self.__remapImage__(edgeImage, corner_list, output_Imageshape)
        maskImage = cv2.warpPerspective(maskImage, prespect_mat, output_Imageshape)
        output_center_list = []
        # cv2.imshow('maskImage', maskImage)
        contours, hier = cv2.findContours(maskImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for con in contours:
            x, y, w, h = cv2.boundingRect(con)
            pixel = maskImage[y + h // 2, x + w // 2]
            color = 2
            if pixel < 100 and pixel > 0:
                color = 1
            output_center_list.append(((x + w // 2, y + h // 2), color))
        # cv2.imshow("dstImage", dstImage)
        return dstImage, output_center_list
        
    def __shadowHist__(self, edgeImage):
        print("\n开始分析棋盘网格直方图...")
        print(f"输入边缘图像尺寸: {edgeImage.shape}")
        
        chess_board_pos = []
        height, width = edgeImage.shape
        
        print("计算水平和垂直投影...")
        x_list = np.sum(edgeImage != 0, axis=0)
        y_list = np.sum(edgeImage != 0, axis=1)
        
        # 计算平均值
        x_aver = np.mean(x_list)
        y_aver = np.mean(y_list)
        print(f"水平投影平均值: {x_aver:.2f}")
        print(f"垂直投影平均值: {y_aver:.2f}")
        
        print("生成投影直方图...")
        x_hist = np.zeros(edgeImage.shape, np.uint8)
        y_hist = np.zeros(edgeImage.shape, np.uint8)
        
        # 绘制直方图
        print("处理投影数据...")
        # 遍历y方向投影数据
        # 确保y_list是numpy数组并转换为整数类型
        y_list = np.asarray(y_list, dtype=np.int32)
        for i in range(y_list.shape[0]):
            val = y_list[i]
            if val > y_aver * 1.3:
                cv2.line(x_hist, (0, i), (val, i), 255, 1)
                
        # 遍历x方向投影数据
        # 确保x_list是numpy数组并转换为整数类型
        x_list = np.asarray(x_list, dtype=np.int32)
        for i in range(x_list.shape[0]):
            val = x_list[i]
            if val > x_aver * 1.3:
                cv2.line(y_hist, (i, 0), (i, val), 255, 1)
        
        print("提取ROI区域...")
        y_roi = y_hist[0:50, 0:width]
        x_roi = x_hist[0:height, 0:50]
        
        print("执行形态学处理...")
        cv2.dilate(x_roi, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)), x_roi)
        cv2.dilate(y_roi, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)), y_roi)
        
        print("提取棋盘网格位置...")
        chess_board_pos = []
        for idx, hist in enumerate([x_hist, y_hist]):
            contours, _ = cv2.findContours(hist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"{'水平' if idx == 0 else '垂直'}方向检测到 {len(contours)} 条线")
            line_pos = []
            
            for contour in contours:
                if idx == 0:  # x方向
                    max_y = max(pt[0][1] for pt in contour)
                    line_pos.append(max_y)
                else:  # y方向
                    max_x = max(pt[0][0] for pt in contour)
                    line_pos.append(max_x)
            
            chess_board_pos.append(sorted(line_pos, reverse=True))
        
        print(f"网格分析完成: 找到 {len(chess_board_pos[0])}x{len(chess_board_pos[1])} 的网格\n")
        return chess_board_pos

    def __remapImage__(self, InputArray, corners, output_size):
        """重映射图像
        Args:
            InputArray: 输入图像
            corners: 四个角点坐标
            output_size: 输出图像大小
        Returns:
            perspective_mat: 透视变换矩阵
            dstImage: 变换后的图像
        """
        try:
            if len(corners) != 4:
                raise ValueError("角点数量必须为4个")
                
            # 构建源点和目标点矩阵
            # 将角点坐标转换为numpy数组
            src_Mat = np.array(corners, dtype=np.float32)
            # 创建目标点矩阵，使用列表推导式避免直接调用np.float32
            dst_points = [
                [0, 0],
                [output_size[0], 0], 
                [output_size[0], output_size[1]],
                [0, output_size[1]]
            ]
            dst_Mat = np.array(dst_points, dtype=np.float32)
            
            # 计算透视变换矩阵
            perspective_mat = cv2.getPerspectiveTransform(src_Mat, dst_Mat)
            if perspective_mat is None:
                raise RuntimeError("透视变换矩阵计算失败")
                
            # 执行透视变换
            dstImage = cv2.warpPerspective(InputArray, perspective_mat, output_size)
            if dstImage is None:
                raise RuntimeError("图像透视变换失败")
                
            return perspective_mat, dstImage
            
        except Exception as e:
            print(f"重映射图像失败: {str(e)}")
            # 返回空矩阵和原始图像
            return np.eye(3), InputArray

    def __position__(self, chess_board_pos, center_list):
        print("\n开始生成棋盘位置矩阵...")
        print(f"网格尺寸: {len(chess_board_pos)}x{len(chess_board_pos[0]) if chess_board_pos else 0}")
        
        # 检查输入数据有效性
        if not chess_board_pos or not chess_board_pos[0]:
            print("错误: 无效的棋盘网格数据")
            return np.zeros((0, 0), np.int)
            
        parser_image = np.zeros([600, 600, 3], np.uint8)
        
        try:
            radius = abs(chess_board_pos[0][0] - chess_board_pos[0][-1]) // len(chess_board_pos[0])
            print(f"计算网格半径: {radius}")
            radius *= 0.7
            
            output_mat = np.zeros([len(chess_board_pos[0]), len(chess_board_pos[1])], np.int)
            chess_position = []
            
            print("生成棋盘坐标...")
            for index, y_val in enumerate(chess_board_pos[0]):
                chess_position.append([])
                for x_val in chess_board_pos[1]:
                    chess_position[index].append((x_val, y_val))
                    
            print("标记棋子位置...")
            for center in center_list:
                color = center[1]
                center = (center[0])
                cv2.circle(parser_image, center, 5, (255 * (color - 1), 255, 0), -1)

            print("分析棋盘交叉点...")
            for index_y, chess_line in enumerate(chess_position):
                for index_x, pos in enumerate(chess_line):
                    cv2.circle(parser_image, pos, 2, (255, 255, 255), -1)
                    cv2.putText(parser_image, f"{index_x},{index_y}", 
                              (pos[0] - 5, pos[1] - 5), 1, 0.5, (255, 0, 0), 1)

                    x1, y1, x2, y2 = pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius
                    for center in center_list:
                        color = center[1]
                        center = center[0]
                        if center[0] > x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                            output_mat[index_y, index_x] = color
                            
            print(f"矩阵生成完成: {output_mat.shape}\n")
            return output_mat
            
        except Exception as e:
            print(f"生成位置矩阵时出错: {str(e)}")
            return np.zeros((0, 0), np.int)
