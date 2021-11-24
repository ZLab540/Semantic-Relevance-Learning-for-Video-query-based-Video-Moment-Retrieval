"""Gets the number of frames and features of each video."""

from absl import app
from absl import flags

import cv2
import glob
import json
import numpy as np
import os

flags.DEFINE_string('feat_dir', 'data/feat/', 'feature directory')

flags.DEFINE_string('video_dir', 'data/videos/', 'video directory')
# 将全部视频下载至'data/videos/'

FLAGS = flags.FLAGS


def main(_):
  # 以只读方式打开文件
  with open('data/activity_net.v1-3.min.json', 'r') as f:
    info = json.load(f)

  all_files = {}
  for i in glob.glob(os.path.join(FLAGS.video_dir, 'v_*')):
    # glob.glob()返回所有匹配的文件路径列表
    name = os.path.split(i)[1]
    # 分离路径和文件名
    name = os.path.splitext(name)[0]
    # 分离文件名和扩展名
    # 此处name为视频名称
    all_files[name[2:]] = i

  data = {}
  for k, v in info['database'].items():
    # items()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回
    # k为视频名称
    feat = np.load(os.path.join(FLAGS.feat_dir, 'v_%s.npy' % k))
    cap = cv2.VideoCapture(all_files[k])
    # 打开视频，参数为视频文件路径
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 得到视频的总帧数
    data[k] = {}
    data[k]['n_frames'] = n_frames
    n_feat = feat.shape[0]
    # 得到矩阵的行数（列数为特征的维数）
    data[k]['n_feat'] = n_feat
    if n_frames // 16 != n_feat:
      print('%s %d %d\t%d' % (
        k, n_feat, n_frames // 16, n_feat - n_frames // 16))
    # //:取整；%d:输出十进制数字；%s:输出字符串；\t:空4个字符
    # 打开一个文件只用于写入。文件存在则将其覆盖，文件不存在则创建新文件
    # 将视频的帧数和特征数写入文件'data/length.json'
  with open('data/length.json', 'w') as f:
    json.dump(data, f)
"""
json.dumps 将一个Python数据结构转换为json
json.loads 将一个json编码的字符串转换回一个python数据结构
json.dump()和json.load()编码和解码json数据，用于处理文件
"""


if __name__ == '__main__':
  app.run(main)
