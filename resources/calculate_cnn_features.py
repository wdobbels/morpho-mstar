'''
In this case, the CNN is imported from a h5 model
'''

import os
import re
import traceback
import datetime
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import keras
from keras.layers import Input, concatenate, Flatten
from keras.applications import VGG16, ResNet50, InceptionResNetV2
from keras import backend as K

chunk_size = 2048
cnntype = 'cnn'  # autoenc or cnn
cnn_name = f'r50_scauto_197_pretrained'  #'r50_scauto_197_gz2', 'xcep_scauto_128_gz2'
if cnntype == 'cnn':
    cnn_model = f'data/cnn_{cnn_name}/model_best.h5'
else:
    cnn_model = f'data/{cnn_name}/model_best.h5'
input_name = 'data/processed_scauto_197.h5'
featsdir = f'./data/cnn_features_{cnn_name}/'  # output (cnn features)
include_top = False
extractionlayer = -3

if not os.path.isdir(featsdir):
    os.makedirs(featsdir)

# Load model (with custom metrics)
print('Loading model...')
def get_accuracy(start, end):
    def task_accuracy(y_true, y_pred):
        return K.cast(K.equal(K.argmax(y_true[:, start:end], axis=-1), 
                              K.argmax(y_pred[:, start:end], axis=-1)), 
                      K.floatx())
    return task_accuracy
t1_acc = get_accuracy(0, 3)
t2_acc = get_accuracy(3, 5)
t3_acc = get_accuracy(5, 7)
t11_acc = get_accuracy(31, 37)
cnn = keras.models.load_model(cnn_model, 
                              custom_objects={'task_accuracy': t1_acc, 'task_accuracy_1': t2_acc,
                                              'task_accuracy_2': t3_acc, 'task_accuracy_3': t11_acc})
if not include_top:
    lastlayer = None
    if cnntype == 'cnn':  # cnn but not last layer
        # pick layer right after global pooling
        # for lay in cnn.layers[::-1]:
        #     if ('global' in lay.name) and ('pooling' in lay.name):
        #         lastlayer = lay
        #         break
        # cnn_out = lastlayer.output
        # pick next to last layer
        lastlayer = cnn.layers[extractionlayer]
        cnn_out = lastlayer.output
    else:  # auto encoder
        for lay in cnn.layers:
            # We need last maxpooling layer
            if 'max_pooling' in lay.name:
                lastlayer = lay
        cnn_out = Flatten()(lastlayer.output)
    if lastlayer is None:
        print(cnn.summary())
        raise ValueError('Did not include top, but could not find last layer!')
    cnn = keras.models.Model(inputs=cnn.inputs, outputs=cnn_out)
            

print('Loading data...')
# Load datafile
f = h5py.File(input_name, 'r')
all_imgs = np.array(f['data'])
all_names = np.array(f['sdss_id'])
ngalaxies = all_imgs.shape[0]
nchunks = ngalaxies // chunk_size

print('nfeats', cnn.output_shape)

nfeats = int(cnn.output_shape[1])
suff = '' if (include_top or (cnntype == 'autoenc')) else '.notop'
if (not include_top) and (extractionlayer < 0):
    suff += f'.layer{abs(extractionlayer)}'
feat_names = list(map(lambda x: cnn_name + '.' + str(x) + suff, range(nfeats)))

# Fix shape
if len(cnn.inputs[0].shape) > len(all_imgs.shape):
    all_imgs = np.expand_dims(all_imgs, axis=-1)
if len(cnn.inputs[0].shape) != len(all_imgs.shape):
    raise ValueError(f'CNN expects input of shape {cnn.inputs[0].shape}, '
                     f'but got input of shape {all_imgs.shape}')

# Preprocessing
if 'r50' in cnn_name:
    print('Preprocessing input...')
    all_imgs = 127.5 * (all_imgs+1.) # from [-1, 1] to [0, 255]
    all_imgs = np.repeat(all_imgs, 3, axis=-1)
    all_imgs[..., 0] -= 103.939
    all_imgs[..., 1] -= 116.779
    all_imgs[..., 2] -= 123.68

def get_start_end(chunkid, totalchunks, length):
    startid = int((length/totalchunks)*chunkid)
    endid = int((length/totalchunks)*(chunkid+1))
    if chunkid > totalchunks - 1:
        return
    if chunkid == totalchunks - 1:
        endid = length
    return startid, endid

print(f'Computing {nfeats} features for {ngalaxies} galaxies in {nchunks} chunks.')
print(f'Starting at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

for chunkid in tqdm(range(nchunks), ascii=True):
    start, end = get_start_end(chunkid, nchunks, ngalaxies)
    cur_chunk_size = end - start
    arr_imgs = all_imgs[start:end, ...]
    names = all_names[start:end]
    # Send through CNN
    arr_feats = np.zeros((cur_chunk_size, nfeats))
    try:
        arr_feats[:, :] = cnn.predict(arr_imgs)
    except ValueError as e:
        allfinite = np.isfinite(arr_feats).all()
        print(f'Error predicting data! img shape: {arr_imgs.shape}. '
              f'Everything finite? {allfinite}')
        if not allfinite:
            badid = np.where(~np.isfinite(arr_feats))
            print(f'Infinite at ID: {badid}')
            print(f'Startid {start}, endid {end}')
        print(traceback.format_exc())
    df_feats = pd.DataFrame(arr_feats, index=names, columns=feat_names)
    df_feats.index.name = 'SDSS_ID'
    df_feats.to_csv(featsdir + f'cnn_feats_{chunkid}.tsv', sep='\t')

print('Finished all chunks. Combining features.')
feat_parts = glob(featsdir+ 'cnn_feats_*.tsv')
feat_frames = []
for feat_part in feat_parts:
    feat_frames.append(pd.read_csv(feat_part, sep='\t', index_col=0))
df_total = pd.concat(feat_frames)
df_total.to_csv(featsdir + 'cnn_feats.tsv', sep='\t')
df_total.to_hdf(featsdir + 'cnn_feats.hdf5', 'cnn_feats')
print(f'Finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')