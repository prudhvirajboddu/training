# LOAD LIBRARIES
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt, cv2
import tensorflow as tf, re, math

# PATHS TO IMAGES
PATH = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/'
PATH2 = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-test/512x512-test/'
IMGS = os.listdir(PATH); IMGS2 = os.listdir(PATH2)
print('There are %i train images and %i test images'%(len(IMGS),len(IMGS2)))

# LOAD TRAIN META DATA
df = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/marking.csv')
df.rename({'image_id':'image_name'},axis=1,inplace=True)
df.head()

# LOAD TEST META DATA
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
test.head()



def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1,feature2):
  feature = {
      'image': _bytes_feature(feature0),
      'image_name': _bytes_feature(feature1),
      'target': _int64_feature(feature2)
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

SIZE = 2071
CT = len(IMGS)//SIZE + int(len(IMGS)%SIZE!=0)
for j in range(CT):
    print(); print('Writing TFRecord %i of %i...'%(j,CT))
    CT2 = min(SIZE,len(IMGS)-j*SIZE)
    with tf.io.TFRecordWriter('train%.2i-%i.tfrec'%(j,CT2)) as writer:
        for k in range(CT2):
            img = cv2.imread(PATH+IMGS[SIZE*j+k])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors
            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            name = IMGS[SIZE*j+k].split('.')[0]
            row = df.loc[df.image_name==name]
            example = serialize_example(
                img, str.encode(name),
                row.patient_id.values[0],
                row.sex.values[0],
                row.age_approx.values[0],                        
                row.anatom_site_general_challenge.values[0],
                row.source.values[0],
                row.target.values[0])
            writer.write(example)
            if k%100==0: print(k,', ',end='')

