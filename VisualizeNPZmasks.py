import matplotlib.pyplot as plt
import numpy as np
import os

npz_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\npz_data\train_npz"
files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

for i in range(3):  # show 3 examples
    data = np.load(os.path.join(npz_dir, files[i]))
    img = data['image']
    mask = data['mask']

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title('Image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title('Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()
