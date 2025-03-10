#!/usr/bin/env python3
import argparse
# 从当前目录导入ChessRecognizer
from ChessRecognizer import ChessRecognizer

def main():
    parser = argparse.ArgumentParser(description='围棋棋盘识别工具')
    parser.add_argument('image_path', help='输入图片路径')
    parser.add_argument('--show', action='store_true', help='显示识别结果')
    parser.add_argument('--save', action='store_true', help='保存识别结果图片')
    args = parser.parse_args()

    try:
        recognizer = ChessRecognizer()
        matrix, _ = recognizer.recognize_from_file(
            args.image_path,
            show_result=args.show,
            save_result=args.save
        )
        print("识别结果矩阵：")
        print(matrix)
        
    except Exception as e:
        print_exception(e)
        return 1
    
    return 0


def print_exception(e, message="操作失败"):
    """打印详细的异常信息"""
    import traceback
    error_stack = traceback.format_exc()
    error_info = traceback.extract_tb(e.__traceback__)[-1]
    
    print(f"\n{message}: {str(e)}")
    print(f"错误位置: {error_info.filename}:{error_info.lineno}")
    print(f"错误函数: {error_info.name}")
    print(f"错误代码: {error_info.line}")
    print(f"错误堆栈:\n{error_stack}")

if __name__ == '__main__':
    exit(main())

