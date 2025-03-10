"""
@Author: Qiangz
@Date: 2019/7/5
@Description:
"""
import tensorflow as tf
from tensorflow.python.framework import graph_util
import argparse

tf.reset_default_graph()  # 重置计算图


def network_structure(args):
    model_path = args.model+'.pb'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        # 获得默认的图
        graph = tf.get_default_graph()
        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            # 得到当前图有几个操作节点
            print("%d ops in the final graph." % len(output_graph_def.node))

            tensor_name = [tensor.name for tensor in output_graph_def.node]
            print(tensor_name)
            print('---------------------------')
            # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
            summaryWriter = tf.summary.FileWriter('log_graph_'+args.model, graph)
            cnt = 0
            for op in graph.get_operations():
                # print出tensor的name和值
                print(op.name, op.values())
                cnt += 1
                if args.n:
                    if cnt == args.n:
                        break


"""
可视化 tensorboard --logdir="log_graph/"
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name to look")
    parser.add_argument('--n', type=int, help='the number of first several tensor name to look') # 当tensor_name过多
    args = parser.parse_args()
    network_structure(args)