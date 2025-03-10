#!/bin/bash

# 查找运行的 Flask 进程
pid=$(lsof -i :5001 | grep Python | awk '{print $2}')

if [ -n "$pid" ]; then
    echo "正在关闭 Flask 服务 (PID: $pid)..."
    kill -9 $pid
    echo "Flask 服务已关闭"
else
    echo "未发现运行中的 Flask 服务"
fi