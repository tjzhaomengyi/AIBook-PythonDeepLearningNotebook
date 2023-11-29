# -*- coding: utf-8 -*-
__author__ = 'Mike'
import os, shutil, pathlib

#生成小批量数据，并给出新的路径
original = "/home/zhaomengyi/Projects/Datas/Kaggle/DogVsCat/PetImages"
new_base = "/home/zhaomengyi/Projects/AIProjects/Book_PythonDeepLearn/chpt8/datas"
original_dir = pathlib.Path(original)
new_base_dir = pathlib.Path(new_base)

def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name
        #os.makedirs(dir)
        fnames = [f'{category}/{i}.jpg' for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copy(src=original_dir / fname, dst=dir/fname)

make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2000)
