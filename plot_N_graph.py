import numpy as np
import matplotlib.pyplot as plt

seqGAN = np.load('.\SeqGan_regular_120_50_200_conv_results.npy')
seqGAN_N = seqGAN[0,2:301]
seqGAN = seqGAN[1,2:301]

benGAN_N = np.load('linf_N.npy')
benGAN = np.load('linf_val.npy')

plt.figure()
plt.semilogy(benGAN_N, benGAN)
plt.semilogy(seqGAN_N, seqGAN)
plt.axhline(y=0.001, color='r', linestyle='--')
plt.xlabel('Number of generator runs (N)', size=16)
plt.ylabel('Error (infinity norm)', size=16)
plt.legend(['Recurrent GAN','seqGAN'])
plt.show()