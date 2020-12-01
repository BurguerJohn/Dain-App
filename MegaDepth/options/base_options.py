import argparse
import os
from ..util import util

class MyOpt():
    dataroot = None
    batchSize = None
    loadSize = None
    fineSize = None
    input_nc = None
    output_nc = None
    ngf = None
    ndf = None
    which_model_netD = None
    which_model_netG = None
    n_layers_D = None
    gpu_ids = None
    name = None
    align_data = None
    model = None
    which_direction = None
    nThreads = None
    checkpoints_dir = None
    norm = None
    serial_batches = None
    display_winsize = None
    display_id = None

    identity = None
    use_dropout = None
    max_dataset_size = None
    isTrain = None


class BaseOptions():
    
    def __init__(self):
        self.opt = MyOpt()
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        
        self.opt.dataroot = True
        self.opt.batchSize = 1
        self.opt.loadSize = 286
        self.opt.fineSize = 256
        self.opt.input_nc = 3
        self.opt.output_nc = 3
        self.opt.ngf = 64
        self.opt.ndf = 64
        self.opt.which_model_netD = 'basic'
        self.opt.which_model_netG = 'unet_256'
        self.opt.n_layers_D = '3'
        self.opt.gpu_ids = '0,1'
        self.opt.name = 'test_local'
        self.opt.align_data = True

        self.opt.model = "pix2pix"
        self.opt.which_direction = "AtoB"
        self.opt.nThreads = "2"
        self.opt.checkpoints_dir = "./checkpoints/"
        self.opt.norm = "instance"
        self.opt.serial_batches = True
        self.opt.display_winsize = 256
        self.opt.display_id = 1

        self.opt.identity = 0.0
        self.opt.use_dropout = True
        self.opt.max_dataset_size = float("inf")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        #self.opt = self.parser.parse_known_args()[0] #parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
