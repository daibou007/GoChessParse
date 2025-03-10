#!/bin/bash
# 设置 TensorFlow 环境变量以启用 v1 兼容模式
export TF_ENABLE_EAGER_EXECUTION=false
export TF_ENABLE_V2_BEHAVIOR=false
python3 GoChessParse.py run
