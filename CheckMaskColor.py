from PIL import Image
import numpy as np

mask_path = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\masks\14_sLea_pyramid [d=3,x=26745,y=23880,w=1536,h=1536].png"  # ← change to one of your actual mask paths
mask = Image.open(mask_path).convert('RGB')
mask_np = np.array(mask)

# Reshape and find unique RGB combinations
unique_colors = np.unique(mask_np.reshape(-1, 3), axis=0)
print("Unique colors in mask:", unique_colors)