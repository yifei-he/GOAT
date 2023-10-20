from PIL import Image
import os
from shutil import copyfile
import numpy as np
import numpy as np
import scipy.io
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


image_options = {
    'batch_size': 100,
    'class_mode': 'binary',
    'color_mode': 'grayscale',
}


def save_data(data_dir='dataset_32x32', save_file='dataset_32x32.mat', target_size=(32, 32)):
    Xs, Ys = [], []
    datagen = ImageDataGenerator(rescale=1./255)
    data_generator = datagen.flow_from_directory(
        data_dir, shuffle=False, target_size=target_size, **image_options)
    while True:
        next_x, next_y = data_generator.next()
        Xs.append(next_x)
        Ys.append(next_y)
        if data_generator.batch_index == 0:
            break
    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    filenames = [f[2:] for f in data_generator.filenames]
    assert(len(set(filenames)) == len(filenames))
    filenames_idx = list(zip(filenames, range(len(filenames))))
    filenames_idx = [(f, i) for f, i in zip(filenames, range(len(filenames)))]
                     # if f[5:8] == 'Cal' or f[5:8] == 'cal']
    indices = [i for f, i in sorted(filenames_idx)]
    genders = np.array([f[:1] for f in data_generator.filenames])[indices]
    binary_genders = (genders == 'F')
    pickle.dump(binary_genders, open('portraits_gender_stats', "wb"))
    print("computed gender stats")
    # gender_stats = utils.rolling_average(binary_genders, 500)
    # print(filenames)
    # sort_indices = np.argsort(filenames)
    # We need to sort only by year, and not have correlation with state.
    # print state stats? print gender stats? print school stats?
    # E.g. if this changes a lot by year, then we might want to do some grouping.
    # Maybe print out number per year, and then we can decide on a grouping? Or algorithmically decide?
    Xs = Xs[indices]
    Ys = Ys[indices]
    scipy.io.savemat('./' + save_file, mdict={'Xs': Xs, 'Ys': Ys})

# Resize images.
def resize(path, size=64):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((size,size), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG')

for folder in ['./dataset_32x32/M/', './dataset_32x32/F/']:
    resize(folder, size=32)

save_data(data_dir='dataset_32x32', save_file='dataset_32x32.mat', target_size=(32,32))