import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', '/ ', 'This is the source directory of the picture')
tf.app.flags.DEFINE_string('output_dir', '/ ', 'This is the saved directory of the picture')
tf.app.flags.DEFINE_integer('test_num', 20, 'This is the number of test of captcha')
tf.app.flags.DEFINE_integer('random_seed', 0, 'This is the random_seed')

#判断tfrecord文件是否存在
def dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        output_filename = os.path.join(dataset_dir,split_name + '.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True

#获取所有验证码图片
def get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path = os.path.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, label0, label1, label2, label3):
    #Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
      'image': bytes_feature(image_data),
      'label0': int64_feature(label0),
      'label1': int64_feature(label1),
      'label2': int64_feature(label2),
      'label3': int64_feature(label3),
    }))

#把数据转为TFRecord格式
def convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']

    with tf.Session() as sess:
        #定义tfrecord文件的路径+名字
        output_filename = os.path.join(dataset_dir,split_name + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()

                    #读取图片
                    image_data = Image.open(filename)  
                    #根据模型的结构resize
                    image_data = image_data.resize((224, 224))
                    #灰度化
                    image_data = np.array(image_data.convert('L'))
                    #将图片转化为bytes
                    image_data = image_data.tobytes()              

                    #获取label
                    labels = filename.split('/')[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))
                                            
                    #生成protocol数据类型
                    example = image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                    
                except IOError as e:
                    print('Could not read:',filename)
                    print('Error:',e)
    sys.stdout.write('\n')
    sys.stdout.flush()

	
def main(unuse_args):
    
    if dataset_exists(FLAGS.output_dir):
        print('tfcecord file has been existed!!')
        
    else:
    #获得所有图片
        photo_filenames = get_filenames_and_classes(FLAGS.dataset_dir)
    #把数据切分为训练集和测试集,并打乱
        random.seed(FLAGS.random_seed)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[FLAGS.test_num:]
        testing_filenames = photo_filenames[:FLAGS.test_num]

    #数据转换
        convert_dataset('train', training_filenames,FLAGS.dataset_dir)
        convert_dataset('test', testing_filenames, FLAGS.dataset_dir)
        print('Finish!!!!!!!!!!!!!!!!!')
	
if __name__ == '__main__':
    tf.app.run()

