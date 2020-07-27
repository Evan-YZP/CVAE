# -*- utf-8 -*-

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers


# now constuct the AxesSubplot objects (in a line)
fig, axs = plt.subplots(1, 2)


with open('reconst.pkl', 'rb') as f:
    reconst_img = pkl.load(f)

frame_num = len(reconst_img)
# clear the scene
def init():
    axs[0].imshow(reconst_img[0][0, :, :, 0])
    axs[0].axis('off') 
    axs[1].imshow(reconst_img[1][0, :, :, 0])
    axs[1].axis('off') 

# read the pre-constructed numpy data
def update(n):
    fig.suptitle(str(n))
    axs[0].clear()
    axs[0].imshow(reconst_img[n][5, :, :, 0])
    axs[0].axis('off') 
    axs[1].clear()
    axs[1].imshow(reconst_img[n][7, :, :, 0])
    axs[1].axis('off') 

ani = FuncAnimation(fig, update, frames=range(frame_num), interval=2000.0/12, save_count=100)

# # save it into gif (need imagemagick)
print('Begin saving gif')
ani.save('test.gif', writer='imagemagick', fps=5)
print('Finished.')

# live show
plt.show()