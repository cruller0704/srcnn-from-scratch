import sys
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sr_convnet import SRConvNet


if len(sys.argv) != 6:
    print('Usage: python3 {} in.pkl in.xxx(image) out.xxx(ground_truth) '
          'out.xxx(bicubic) out.xxx(srcnn)'.format(sys.argv[0]))
    sys.exit(1)

fnp = sys.argv[1]
fni = sys.argv[2]
fng = sys.argv[3]
fnb = sys.argv[4]
fns = sys.argv[5]

num_ch = 1
patch_size = 33
scale = 3

im_orig = Image.open(fni)
im_orig = im_orig.convert('YCbCr')
height = im_orig.height
width = im_orig.width
im_orig.show()
im_orig.convert('RGB').save(fng)
im_aorig = np.array(im_orig)/255

im = im_orig.resize((width//scale, height//scale), resample=Image.BICUBIC)
im = im.resize((width, height), resample=Image.BICUBIC)
im.show()
im.convert('RGB').save(fnb)
im_bi = np.array(im).transpose(2, 0, 1)/255
psnr_bi = -10*np.log10(np.sum((im_bi[0, :, :] -
                               im_aorig[:, :, 0])**2)/(height*width))
print('Bicubic:', psnr_bi)
im_in = np.empty((1, num_ch, height, width))
im_in[0] = im_bi[0, :, :]

network = SRConvNet(input_dim=(num_ch, patch_size, patch_size),
                    conv_param_1={'filter_num': 64, 'filter_size': 9,
                                  'pad': 4, 'stride': 1},
                    conv_param_2={'filter_num': 32, 'filter_size': 1,
                                  'pad': 0, 'stride': 1},
                    conv_param_3={'filter_num': 1, 'filter_size': 5,
                                  'pad': 2, 'stride': 1})
network.load_params(fnp)
im_out = network.predict(im_in)

im_mux = np.empty((height, width, 3))
im_mux[:, :, 0] = np.clip(im_out[0, 0], 0, 1)
im_mux[:, :, 1:] = im_bi[1:].transpose(1, 2, 0)
psnr_sr = -10*np.log10(np.sum((im_mux[:, :, 0] -
                               im_aorig[:, :, 0])**2)/(height*width))
print('SRCNN:  ', psnr_sr)
im_mux = (im_mux*255).astype(np.uint8)
im_sr = Image.fromarray(im_mux, 'YCbCr')
im_sr.show()
# plt.imshow(np.array(im_mux), interpolation='nearest')
# plt.show()

im_sr.convert('RGB').save(fns)
