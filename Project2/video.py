import numpy as np
import matplotlib.pyplot as plt
import os
import moviepy.video.io.ImageSequenceClip

image_folder='images'
fps=10

image_files = ['images/test'+str(i)+'.png' for i in range(1000)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('my_video.mp4')
