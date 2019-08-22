import sys
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sr_convnet import SRConvNet


if len(sys.argv) != 4:
    print('Usage: python3 {} in.pkl in.xxx(image) '
          'out.xxx(image)'.format(sys.argv[0]))
    sys.exit(1)

fnp = sys.argv[1]
fni = sys.argv[2]
fno = sys.argv[3]

num_ch = 1
patch_size = 33
scale = 3

im = Image.open(fni)
im = im.resize((im.width*scale, im.height*scale), resample=Image.BICUBIC)
im = im.convert('YCbCr')
height = im.height
width = im.width
im.show()

im_bi = np.array(im).transpose(2, 0, 1)/255
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
im_mux = (im_mux*255).astype(np.uint8)
im_sr = Image.fromarray(im_mux, 'YCbCr')
im_sr.show()
# plt.imshow(np.array(im_mux), interpolation='nearest')
# plt.show()

im_sr.convert('RGB').save(fno)
