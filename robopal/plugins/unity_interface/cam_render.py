import numpy as np

from mjremote import mjremote
import time
import cv2

m = mjremote()
print('Connect: ', m.connect())
b = bytearray(3 * m.width * m.height)
t0 = time.time()
m.getimage(b)

t1 = time.time()
print('FPS: ', (t1-t0))
m.close()
b = np.asarray(b).reshape(m.height, m.width, 3)
cv2.imshow("test", b)
cv2.waitKey(0)
