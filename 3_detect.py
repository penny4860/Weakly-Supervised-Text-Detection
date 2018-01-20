
import matplotlib.pyplot as plt
from src.net import text_activation_map

text_map, im = text_activation_map("weights//weights.16.h5", "images//1.png")

fig, ax = plt.subplots()
ax.imshow(im, alpha=0.5)
ax.imshow(text_map, cmap='jet', alpha=0.5)
plt.show()

