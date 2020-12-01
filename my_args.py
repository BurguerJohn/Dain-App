import os
import datetime
import argparse
import numpy
import networks
import  torch
modelnames =  networks.__all__
# import datasets
datasetNames = ('Vimeo_90K_interp') #datasets.__all__

parser = argparse.ArgumentParser(description='DAIN')

parser.add_argument('--debug',action = 'store_true', help='Enable debug mode')
parser.add_argument('--netName', type=str, default='DAIN',
                    choices = modelnames,help = 'model architecture: ' +
                        ' | '.join(modelnames) +
                        ' (default: DAIN)')

parser.add_argument('--datasetName', default='Vimeo_90K_interp',
                    choices= datasetNames,nargs='+',
                    help='dataset type : ' +
                        ' | '.join(datasetNames) +
                        ' (default: Vimeo_90K_interp)')
parser.add_argument('--datasetPath',default='',help = 'the path of selected datasets')
parser.add_argument('--dataset_split', type = int, default=97, help = 'Split a dataset into trainining and validation by percentage (default: 97)')

parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

parser.add_argument('--numEpoch', '-e', type = int, default=100, help= 'Number of epochs to train(default:150)')

parser.add_argument('--batch_size', '-b',type = int ,default=1, help = 'batch size (default:1)' )
parser.add_argument('--workers', '-w', type =int,default=8, help = 'parallel workers for loading training samples (default : 1.6*10 = 16)')
parser.add_argument('--channels', '-c', type=int,default=3,choices = [1,3], help ='channels of images (default:3)')
parser.add_argument('--filter_size', '-f', type=int, default=4, help = 'the size of filters used (default: 4)',
                    choices=[2,4,6, 5,51]
                    )


parser.add_argument('--lr', type =float, default= 0.002, help= 'the basic learning rate for three subnetworks (default: 0.002)')
parser.add_argument('--rectify_lr', type=float, default=0.001, help  = 'the learning rate for rectify/refine subnetworks (default: 0.001)')

parser.add_argument('--save_which', '-s', type=int, default=1, choices=[0,1], help='choose which result to save: 0 ==> interpolated, 1==> rectified')
parser.add_argument('--time_step',  type=float, default=0.5, help='choose the time steps')
parser.add_argument('--flow_lr_coe', type = float, default=0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')
parser.add_argument('--occ_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--filter_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--ctx_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--depth_lr_coe', type = float, default=0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')

parser.add_argument('--alpha', type=float,nargs='+', default=[0.0, 1.0], help= 'the ration of loss for interpolated and rectified result (default: [0.0, 1.0])')

parser.add_argument('--epsilon', type = float, default=1e-6, help = 'the epsilon for charbonier loss,etc (default: 1e-6)')
parser.add_argument('--weight_decay', type = float, default=0, help = 'the weight decay for whole network ' )
parser.add_argument('--patience', type=int, default=5, help = 'the patience of reduce on plateou')
parser.add_argument('--factor', type = float, default=0.2, help = 'the factor of reduce on plateou')

parser.add_argument('--pretrained', dest='SAVED_MODEL', default=None, help ='path to the pretrained model weights')
parser.add_argument('--no-date', action='store_true', help='don\'t append date timestamp to folder' )
parser.add_argument('--use_cuda', default= True, type = bool, help='use cuda or not')
parser.add_argument('--use_cudnn',default=1,type=int, help = 'use cudnn or not')
parser.add_argument('--dtype', default=torch.cuda.FloatTensor, choices = [torch.cuda.FloatTensor,torch.FloatTensor],help = 'tensor data type ')
# parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')


parser.add_argument('--uid', type=str, default= None, help='unique id for the training')
parser.add_argument('--force', action='store_true', help='force to override the given uid')

parser.add_argument('--video', type = str, default= None, help='')
parser.add_argument('--outStr', type = str, default= None, help='')
parser.add_argument('--outFolder', type = str, default= None, help='')
parser.add_argument('--fps', type = float, default= None, help='')
parser.add_argument('--palette', type = int, default= 0, help='')
parser.add_argument('--resc', type = int, default= 0, help='')
parser.add_argument('--maxResc', type = int, default= 0, help='')
parser.add_argument('--loop', type = int, default= 0, help='')
parser.add_argument('--framerateConf', type = int, default= 0, help='')
parser.add_argument('--use60RealFps', type = float, default= 60, help='')
parser.add_argument('--use60', type = int, default= 0, help='')
parser.add_argument('--use60C1', type = int, default= 0, help='')
parser.add_argument('--use60C2', type = int, default= 0, help='')
parser.add_argument('--interpolationMethod', type = int, default= 0, help='')
parser.add_argument('--exportPng', type = int, default= 0, help='')
parser.add_argument('--useAnimationMethod', type = int, default= 1, help='')
parser.add_argument('--splitFrames', type = int, default= 0, help='')
parser.add_argument('--splitSize', type = int, default= 0, help='')
parser.add_argument('--splitPad', type = int, default= 0, help='')
parser.add_argument('--alphaMethod', type = int, default= 0, help='')
parser.add_argument('--inputMethod', type = int, default= 0, help='')
parser.add_argument('--cleanOriginal', type = int, default= 1, help='')
parser.add_argument('--cleanInterpol', type = int, default= 1, help='')
parser.add_argument('--doOriginal', type = int, default= 1, help='')
parser.add_argument('--doIntepolation', type = int, default= 1, help='')
parser.add_argument('--doVideo', type = int, default= 1, help='')
parser.add_argument('--checkSceneChanges', type = int, default= 1, help='')
parser.add_argument('--sceneChangeSensibility', type = int, default= 10, help='')

parser.add_argument('--uploadBar', type = None, default= None, help='')
parser.add_argument('--useWatermark', type = int, default= 0, help='')

args = parser.parse_args()

import shutil


save_path = ""
parser.add_argument('--save_path',default=save_path,help = 'the output dir of weights')
parser.add_argument('--log', default = save_path+'/log.txt', help = 'the log file in training')
parser.add_argument('--arg', default = save_path+'/args.txt', help = 'the args used')

args = parser.parse_args()



