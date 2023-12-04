# -*- coding: utf-8 -*-
__author__ = 'Mike'
import os, pathlib, shutil, random
'''
1、将20%训练文本文件放入一个新的目录，acImdb/val
'''
imdb_path = "/home/zhaomengyi/Projects/Datas/IMDB/aclImdb_v1/aclImdb/"
base_dir = pathlib.Path(imdb_path)
val_dir = base_dir / "val"
train_dir = base_dir / "train"
for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir/ category / fname,
                    val_dir/ category / fname)
