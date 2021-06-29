import os
import pandas as pd
import shutil

csv_path='train.csv'

rd=pd.read_csv(csv_path)

folder = os.listdir('train')

train_dir = 'train_dir'
os.mkdir(train_dir)

os.path.join(train_dir,'non_melanoma')
os.path.join(train_dir,'melanoma')

for row in rd.iterrows():
    image = row[1].image_name
    label = row[1].target
    fname = image + '.jpg'
    if fname in folder:
        src = os.path.join('train', fname)
        dst = os.path.join(str(label),fname)
        shutil.copyfile(src, dst)

