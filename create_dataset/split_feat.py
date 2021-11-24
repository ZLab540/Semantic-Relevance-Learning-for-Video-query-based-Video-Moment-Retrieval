"""Subsample the features so that there is no overlap."""

from absl import app
from absl import flags

import h5py
import numpy as np
import os

flags.DEFINE_string('feat_dir', 'data/feat/', 'feature directory')

FLAGS = flags.FLAGS


def main(_):
  f = h5py.File('data/sub_activitynet_v1-3.c3d.hdf5', 'r')

  if not os.path.exists(FLAGS.feat_dir):
    os.mkdir(FLAGS.feat_dir)
    # 创建目录

  for i in f.keys():
    feat = f[i + '/c3d_features'][:]
    feat = feat[::2].astype(np.float32)
    # 取'0 2 4 6......'处的数据
    # C3D每16帧提取特征(重叠率为50%，即1-16、9-24、17-32......)
    # 此处选取的特征为(1-16、17-32......)
    np.save(os.path.join(FLAGS.feat_dir, i), feat)
    # np.save(file,arr,allow_pickle=True,fix_imports=True)
    # file:要保存的文件名称，文件扩展名为.npy
    # arr:需要保存的数组


if __name__ == '__main__':
  app.run(main)
