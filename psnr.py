import PIL
import PIL.Image
import numpy as np

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    i1 = PIL.Image.open(img1).convert('RGB')
    i2 = PIL.Image.open(img2).convert('RGB')

    mse = np.mean((np.array(i1, dtype=np.float32) - np.array(i2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def IsDiffScenes( img1, img2, diff=25):
  return calculate_psnr(img1, img2) <= diff