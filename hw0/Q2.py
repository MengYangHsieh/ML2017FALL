from PIL import Image
from sys import argv
im = Image.open(argv[1])
out = im.point(lambda i : i // 2)
out.save("Q2.png")