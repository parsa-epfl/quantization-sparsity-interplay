import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors


head_idx = 0
PATH_TO_FOLDER = "/home/parsa_liza/experiments/kqv_distributions/"
kqv_fp_dict = pickle.load(open(PATH_TO_FOLDER + "fp-last_layer.pkl", "rb"))
kqv_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "fp-sparse-last_layer.pkl", "rb"))
kqv_hbfp_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp-last_layer.pkl", "rb"))
kqv_hbfp_sparse_dict = pickle.load(open(PATH_TO_FOLDER + "hbfp-sparse-last_layer.pkl", "rb"))

error_key_sparse = abs(kqv_fp_dict["key"][head_idx] - kqv_sparse_dict["key"][head_idx])[:200, :]
error_key_hbfp = abs(kqv_fp_dict["key"][head_idx] - kqv_hbfp_dict["key"][head_idx])[:200, :]
error_key_hbfp_sparse = abs(kqv_fp_dict["key"][head_idx] - kqv_hbfp_sparse_dict["key"][head_idx])[:200, :]


#fig, axis = plt.subplots(1, 3)
#axis[0].imshow(error_key_hbfp)
#scale_img = axis[1].imshow(error_key_sparse)
#axis[2].imshow(error_key_hbfp_sparse)
#fig.colorbar(scale_img, ax=axs, orientation='horizontal', fraction=.1)
#fig.suptitle("Error magnitudes for key projections: layer 0 head " + str(head_idx))
#plt.savefig(PATH_TO_FOLDER + 'errors.png')



Nr = 1
Nc = 3

fig, axs = plt.subplots(Nr, Nc)
fig.suptitle("Error magnitudes for key projections: layer 12 head " + str(head_idx), fontsize="large")

images = []
error_values = [error_key_hbfp, error_key_sparse, error_key_hbfp_sparse]
error_names = ["bfp8_dense", "fp32_sparse", "bfp8_sparse"]
for j in range(Nc):
    data = error_values[j]
    images.append(axs[j].imshow(data))
    axs[j].set_title(error_names[j], fontsize="medium")
    axs[j].label_outer()

# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

# fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, ticks=np.arange(0, 1.4, 0.1))
fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

# Make images respond to changes in the norm of other images (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# recurse infinitely!
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


for im in images:
    im.callbacks.connect('changed', update)

plt.savefig(PATH_TO_FOLDER + 'key_layer12_head' + str(head_idx) + '.png')