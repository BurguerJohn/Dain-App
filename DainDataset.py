import torch.utils.data as data
import torch
from torch.autograd import Variable
import numpy
import PIL
import PIL.Image
import psnr
import RenderData

class DainDataset(data.Dataset):
  def __init__(self, my_list, pad, diffScenes = -1, frameFormat = "RGB", addPadding = True, useHalf = False):
    self.list = my_list
    self.pad = pad
    self.combos = []
    self.addPad = addPadding
    self.frameFormat = frameFormat
    self.useHalf = useHalf


    for i in range(0, len(my_list) - 1):
      if diffScenes > -1:
        skip_interpolation = psnr.IsDiffScenes(my_list[i], my_list[i + 1], diffScenes)
        if skip_interpolation:
          print("Scene detection between frames {} and {}".format(i ,i+1))
          continue
      self.combos.append({"f1": my_list[i], "f2": my_list[i + 1], "i" : -1})

  def Convert(self, a):
    return torch.FloatTensor(a.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
  def Prepare(self, X):
    X = Variable(torch.unsqueeze(X,0))
    if self.addPad:
      X = torch.nn.functional.pad(X, (self.pad[0], self.pad[1] , self.pad[2], self.pad[3]), mode='replicate', value=0)
    if self.useHalf:
      X = X.half()
    return X[0]

  def __getitem__(self, index):
    c1 = PIL.Image.open(self.combos[index]['f1']).convert(self.frameFormat)
    c1 = self.Convert(numpy.array(c1))
    c1 = self.Prepare(c1)

    c2 = PIL.Image.open(self.combos[index]['f2']).convert(self.frameFormat)
    c2 = self.Convert(numpy.array(c2))
    c2 = self.Prepare(c2)


    my_combo = self.combos[index]
    my_combo["original"] = numpy.array(PIL.Image.open(self.combos[index]["f1"]).convert(self.frameFormat))

    return (my_combo, c1, c2)

  def __len__(self):
    return len(self.combos)