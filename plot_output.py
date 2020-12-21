import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('outputs/output_multipleruns.pickle', 'rb') as f:
    output = pickle.load(f)

aucs_vamp = np.zeros((len(output.keys()), 5))
aucs_stand = np.zeros((len(output.keys()), 5))
val_loss_vamp = np.zeros(len(output.keys()))
val_loss_stand = np.zeros(len(output.keys()))

for i, key in enumerate(output.keys()):
    aucs_vamp[i,:] = np.mean(output[key]['aucs'][1], axis = 0)
    aucs_stand[i,:] = np.mean(output[key]['aucs'][0], axis =0)
    val_loss_vamp[i] = output[key]['validation_loss'][1]
    val_loss_stand[i] = output[key]['validation_loss'][0]


plt.plot(list(output.keys()), aucs_vamp, label = ('Elbo grad', 'Recon grad', 'KL grad', 'Reconstruction error'))
plt.legend()
plt.show()

plt.plot(list(output.keys()), aucs_stand, label = ('Elbo grad', 'Recon grad', 'KL grad', 'Reconstruction error'))
plt.legend()
plt.show()

plt.plot(list(output.keys()), val_loss_vamp, label = 'Vamp prior')
plt.plot(list(output.keys()), val_loss_stand, label = 'Standard gaussian prior')
plt.legend()
plt.show()

