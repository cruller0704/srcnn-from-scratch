import glob
import sys
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cnn.trainer import Trainer
from sr_convnet import SRConvNet


if len(sys.argv) != 2:
    print('Usage: python3 {} out.pkl'.format(sys.argv[0]))
    sys.exit(1)

fno = sys.argv[1]
# N = int(sys.argv[hoge])

num_train = 1024
num_test = 128
num_ch = 1
patch_size = 33
scale = 3
t_train = np.empty((num_train, num_ch, patch_size, patch_size))
x_train = np.empty((num_train, num_ch, patch_size, patch_size))
t_test = np.empty((num_test, num_ch, patch_size, patch_size))
x_test = np.empty((num_test, num_ch, patch_size, patch_size))

fnr = glob.glob('./dataset/train/*.png')
fne = glob.glob('./dataset/test/*.png')
num_file_train = len(fnr)
num_file_test = len(fne)

print('Sampling dataset...')
fnr_idx = np.random.randint(num_file_train, size=num_train)
n = 0
for i in range(num_file_train):
    num_train_i = len(fnr_idx[fnr_idx == i])
    im = Image.open(fnr[i])
    for j in range(num_train_i):
        x = np.random.randint(im.size[0] - patch_size + 1)
        y = np.random.randint(im.size[1] - patch_size + 1)
        patch = im.crop((x, y, x + patch_size, y + patch_size))
        patch = patch.convert('YCbCr')
        t_train[n] = np.array(patch).transpose(2, 0, 1)[0]/255

        patch = patch.resize((patch_size//scale, patch_size//scale),
                             resample=Image.BICUBIC)
        patch = patch.resize((patch_size, patch_size),
                             resample=Image.BICUBIC)
        x_train[n] = np.array(patch).transpose(2, 0, 1)[0]/255
        # plt.imshow(x_train[n, 0], cmap=plt.cm.gray,
        #            interpolation='nearest')
        # plt.show()
        # sys.exit(0)
        n += 1

fne_idx = np.random.randint(num_file_test, size=num_test)
n = 0
for i in range(num_file_test):
    num_test_i = len(fne_idx[fne_idx == i])
    im = Image.open(fne[i])
    for j in range(num_test_i):
        x = np.random.randint(im.size[0] - patch_size + 1)
        y = np.random.randint(im.size[1] - patch_size + 1)
        patch = im.crop((x, y, x + patch_size, y + patch_size))
        patch = patch.convert('YCbCr')
        t_test[n] = np.array(patch).transpose(2, 0, 1)[0]/255

        patch = patch.resize((patch_size//scale, patch_size//scale),
                             resample=Image.BICUBIC)
        patch = patch.resize((patch_size, patch_size),
                             resample=Image.BICUBIC)
        x_test[n] = np.array(patch).transpose(2, 0, 1)[0]/255
        # plt.imshow(x_test[n, 0], cmap=plt.cm.gray,
        #            interpolation='nearest')
        # plt.show()
        # sys.exit(0)
        n += 1
print('Done.')

network = SRConvNet(input_dim=(num_ch, patch_size, patch_size),
                    conv_param_1={'filter_num': 64, 'filter_size': 9,
                                  'pad': 0, 'stride': 1},
                    conv_param_2={'filter_num': 32, 'filter_size': 1,
                                  'pad': 0, 'stride': 1},
                    conv_param_3={'filter_num': 1, 'filter_size': 5,
                                  'pad': 0, 'stride': 1})
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=2000, mini_batch_size=64, optimizer='SRCNNOrig',
                  optimizer_param={'lr': 1e-4, 'lrl': 1e-5},
                  evaluate_sample_num_per_epoch=128)
trainer.train()

# Save parameters
network.save_params(fno)
print("Saved Network Parameters!")
