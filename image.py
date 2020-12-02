import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

#src=cv2.imread('/home/radhakumaran/Downloads/tetris_blocks.png') # Source image

src = mpimg.imread("/home/radhakumaran/Downloads/tetris_blocks.png")
plt.subplot(211),plt.imshow(src),plt.title('image')
plt.xticks([]),plt.yticks([])
#plt.axis("off")

plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imshow('image',src)
cv2.waitKey(0)
cv2.destroyWindow()
