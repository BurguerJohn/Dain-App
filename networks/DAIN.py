# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

from my_package.FilterInterpolation import  FilterInterpolationModule
from my_package.FlowProjection import  FlowProjectionModule #,FlowFillholeModule
from my_package.DepthFlowProjection import DepthFlowProjectionModule

from Stack import Stack

import PWCNet
import S2D_models
import Resblock
import MegaDepth
import time
import RenderData
import cv2
import numpy as np
import time
import datetime as dt

from MotionBlur import BlurIt


SHOW_TIMING = False
TIMER = None
def SetTiming(tag):
  global TIMER, SHOW_TIMING
  if TIMER == None:
    TIMER = time.time()
    if SHOW_TIMING:
      print("\nStarting DAIN Timer: " + tag)
  if SHOW_TIMING:
    torch.cuda.synchronize()
    print("\nDAIN "+tag + ": --- %s  seconds ---" % (time.time() - TIMER))
    TIMER = time.time()



class DAIN(torch.nn.Module):
    def __init__(self,
                 channel = 3,
                 filter_size = 4,
                 timestep=0.5,
                 training=True,
                 useAnimationMethod = 0,
                 shareMemory = False,
                 cuda = True, iSize = [2048, 1024], batch_size = 1, padding = (0,0,0,0),
                 depadding = (0,0,0,0),
                 is_half = False):

        # base class initialization
        super(DAIN, self).__init__()
        
        
        self.filter_size = filter_size
        self.training = training
        self.timestep = timestep
        self.useAnimationMethod = useAnimationMethod
        assert (timestep == 0.5) # TODO: or else the WeigtedFlowProjection should also be revised... Really Tedious work.
        self.numFrames =int(1.0/timestep) - 1
        self.use_cuda = cuda
        self.iSize = iSize
        self.padding = padding
        self.depadding = depadding
        self.is_half = is_half

        i=0
        self.initScaleNets_filter,self.initScaleNets_filter1,self.initScaleNets_filter2 = \
            self.get_MonoNet5(channel if i == 0 else channel + filter_size * filter_size, filter_size * filter_size, "filter")

        self.ctxNet = S2D_models.__dict__['S2DF_3dense']()

        self.ctx_ch = 3 * 64 + 3

        self.rectifyNet = Resblock.__dict__['MultipleBasicBlock_4'](3 + 3 + 3 +2*1+ 2*2 +16*2+ 2 * self.ctx_ch, 128)

        if self.is_half:
            self.ctxNet = self.ctxNet.half()
            self.rectifyNet = self.rectifyNet.half()
            self.initScaleNets_filter = self.initScaleNets_filter.half()
            self.initScaleNets_filter1 = self.initScaleNets_filter1.half()
            self.initScaleNets_filter2 = self.initScaleNets_filter2.half()
        else:
            self.ctxNet = self.ctxNet.float()
            self.rectifyNet = self.rectifyNet.float()
            self.initScaleNets_filter = self.initScaleNets_filter.float()
            self.initScaleNets_filter1 = self.initScaleNets_filter1.float()
            self.initScaleNets_filter2 = self.initScaleNets_filter2.float()

        if self.use_cuda:
            self.ctxNet = self.ctxNet.cuda()
            self.initScaleNets_filter = self.initScaleNets_filter.cuda()
            self.initScaleNets_filter1 = self.initScaleNets_filter1.cuda()
            self.initScaleNets_filter2 = self.initScaleNets_filter2.cuda()
            self.rectifyNet = self.rectifyNet.cuda()
        else:
            self.ctxNet = self.ctxNet.cpu()
            self.initScaleNets_filter = self.initScaleNets_filter.cpu()
            self.initScaleNets_filter1 = self.initScaleNets_filter1.cpu()
            self.initScaleNets_filter2 = self.initScaleNets_filter2.cpu()
            self.rectifyNet = self.rectifyNet.cpu()


        #self._initialize_weights()

        if self.training:
            self.flownets = PWCNet.__dict__['pwc_dc_net']("PWCNet/pwc_net.pth.tar")
        else:
            self.flownets = PWCNet.__dict__['pwc_dc_net'](None, self.use_cuda, iSize, batch_size)

        if self.is_half:
            self.flownets = self.flownets.half()
        else:
            self.flownets = self.flownets.float()

        if self.use_cuda:
            self.flownets = self.flownets.cuda()
        else:
            self.flownets = self.flownets.cpu()


        self.div_flow = 20.0

        if useAnimationMethod == 0:
            #extract depth information
            if self.training:
                self.depthNet=MegaDepth.__dict__['HourGlass']("MegaDepth/checkpoints/test_local/best_generalization_net_G.pth")
            else:
                self.depthNet=MegaDepth.__dict__['HourGlass']()
            
            if self.is_half:
                self.depthNet = self.depthNet.half()
            else:
                self.depthNet = self.depthNet.float()
            
            if self.use_cuda:
                self.depthNet=self.depthNet.cuda()
            else:
                self.depthNet=self.depthNet.cpu()


        self.filterModule = FilterInterpolationModule.__dict__['interpolation']()

        if self.is_half:
            self.filterModule = self.filterModule.half()

        if self.use_cuda:
            self.filterModule = self.filterModule.cuda()
        else:
            self.filterModule = self.filterModule.cpu()

        return

    def _initialize_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                count+=1
                nn.init.xavier_uniform_(m.weight.data)
                # weight_init.kaiming_uniform(m.weight.data, a = 0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            # else:
            #     print(m)


    def forward(self, input_0, input_2, padding, flow_force, flow_smooth, flow_share, convert, rectify):

        SetTiming("Starting Foward")

        losses = []
        offsets= []
        filters = []
        occlusions = []


        alpha = False       

        torch.cuda.synchronize()
        if input_0.size(1) == 4:
            alpha = True

            #print("There is an alpha channel")
            input_0_alpha = input_0[:,3:4,:,:]
            input_0 = input_0[:,0:3,:,:]
            input_2_alpha = input_2[:,3:4,:,:]
            input_2 = input_2[:,0:3,:,:]

            if self.use_cuda:
                input_0_alpha = input_0_alpha.cuda()
                input_0 = input_0.cuda()
                input_2_alpha = input_2_alpha.cuda()
                input_2 = input_2.cuda()

            
            if self.is_half:
                input_0_alpha = input_0_alpha.half()
                input_2_alpha = input_2_alpha.half()
                input_0 = input_0.half()
                input_2 = input_2.half()

        elif input_0.size(1) == 3:
            if self.use_cuda:
                input_0 = input_0.cuda()
                input_2 = input_2.cuda()

            if self.is_half:
                input_0 = input_0.half()
                input_2 = input_2.half()

        else:
            print("Bug??")

        torch.cuda.synchronize()

        downscaler = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        upscaler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        SetTiming("Starting Cat")

        SetTiming("Ending Cat / Start Depth")

        if self.useAnimationMethod == 1 or self.useAnimationMethod == 2:
            #print(str(input_0.shape) +" "+  str(input_2.shape))
            temp = torch.cat((input_0, input_2),dim=0)
            temp = temp[:, 1:2, :, :]
        else:
            torch.cuda.synchronize()
            temp = self.depthNet(torch.cat((input_0, input_2),dim=0))
            if self.is_half:
                temp *= -1
        
        
        log_depth = [temp[:input_0.size(0)], temp[input_2.size(0):]]
        SetTiming("End Depth / Start Filter 01")

        if self.useAnimationMethod == 1:
            log_depth[0].fill_(1)
            log_depth[1].fill_(1)
        if self.useAnimationMethod == 2:
            log_depth = [d  for d in log_depth]

        if self.useAnimationMethod == 1:
            depth_inv = log_depth 
            #depth_inv = [None , None]         
        else:
            if self.is_half:
                depth_inv = [(1e-6 + 1 / torch.exp(d.float())).half() for d in log_depth]
            else:
                depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]

        SetTiming("End Calculations")

        s1 = torch.cuda.current_stream()
        s2 = torch.cuda.current_stream()
        
        fSingle = self.forward_singlePath(self.initScaleNets_filter, torch.cat((input_0, input_2), dim=1), 'filter')
        with torch.cuda.stream(s1):
            
            cur_offset_outputs0 = self.forward_flownets(self.flownets, input_0, input_2, 0.5, 20, 0)
            
            offset1 = self.FlowProject(cur_offset_outputs0, depth_inv[0])[0]
            if flow_smooth == 0:
                del cur_offset_outputs0

            filter1 = self.forward_singlePath(self.initScaleNets_filter1, fSingle, name=None)
           
            if rectify:
                cur_ctx_output0 = torch.cat((self.ctxNet(input_0), log_depth[0]), dim=1)
                ctx0 = self.filterModule(cur_ctx_output0, offset1, filter1)
                del cur_ctx_output0
            ref0 = self.filterModule(input_0, offset1, filter1) * 0.5
            if alpha:
                alpha0 = self.filterModule(input_0_alpha *1.1 , offset1, filter1) * 0.5
        with torch.cuda.stream(s2):
            
            cur_offset_outputs1 = self.forward_flownets(self.flownets, input_2, input_0, 0.5, 20, 0) 
            
            offset2 = self.FlowProject(cur_offset_outputs1,depth_inv[1])[0]
            if flow_smooth == 0:
                del cur_offset_outputs1

            filter2 = self.forward_singlePath(self.initScaleNets_filter2, fSingle, name=None)
            
            if rectify:
                cur_ctx_output1 = torch.cat((self.ctxNet(input_2), log_depth[1]), dim=1)
                ctx2 = self.filterModule(cur_ctx_output1, offset2, filter2)
                del cur_ctx_output1
            ref2 = self.filterModule(input_2, offset2, filter2) * 0.5
            if alpha:
                alpha2 = self.filterModule(input_2_alpha *1.1 , offset2, filter2) * 0.5

        torch.cuda.synchronize()

        del fSingle
        cur_output =  ref0 + ref2
        if alpha:
            alpha_output = (alpha0 + alpha2).cpu().float()

        if not rectify:
            return [None, cur_output],None,None

        del log_depth
        del depth_inv
        del input_0
        del input_2

        SetTiming("End forward_singlePath 2")


        torch.cuda.synchronize()

        if flow_smooth != 0:
            one = torch.abs(cur_offset_outputs0[0][:, 0, :,:]) + torch.abs(cur_offset_outputs0[0][:, 1, :,:])
            two = torch.abs(cur_offset_outputs1[0][:, 0, :,:]) + torch.abs(cur_offset_outputs1[0][:, 1, :,:])

            mem1 = cur_offset_outputs0[0][:, :, :,:]
            mem2 = cur_offset_outputs1[0][:, :, :,:]

            one -= 1
            two -= 1

            one = torch.clamp(one, min = 0, max=10) / 10
            two = torch.clamp(two, min = 0, max=10) / 10

            bigger = torch.max(one, two)
            del two

            bigger = bigger.unsqueeze(0)
            one = one.unsqueeze(0)


        SetTiming("End Flow 01 / Start Flow 02")

        depadding = self.depadding

        if alpha and convert:
            alpha_output  = alpha_output[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]

        cur_output  = cur_output[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]
        ref0        = ref0[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]
        ref2        = ref2[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]
        offset1     = offset1[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]
        offset2     = offset2[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]
        filter1     = filter1[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]
        filter2     = filter2[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]
        ctx0        = ctx0[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]
        ctx2        = ctx2[:, :, depadding[2]:depadding[3], depadding[0]: depadding[1]]


        rectify_input = torch.cat((cur_output, ref0),dim =1)
        del ref0
        rectify_input = torch.cat((rectify_input, ref2),dim =1)
        del ref2
        rectify_input = torch.cat((rectify_input, offset1, offset2),dim =1)
        del offset1
        del offset2
        rectify_input = torch.cat((rectify_input, filter1, filter2),dim =1)
        del filter1
        del filter2
        rectify_input = torch.cat((rectify_input, ctx0),dim =1)
        del ctx0
        rectify_input = torch.cat((rectify_input, ctx2),dim =1)
        del ctx2
       
        torch.cuda.synchronize()
        SetTiming("End Clean Cache / Start Rectify")
        
        
        cur_output_rectified = self.rectifyNet(rectify_input) + cur_output

        SetTiming("End Rectify / Start Ending")
        
        cur_output_rectified = cur_output_rectified.cpu().float()
 
        if not convert:
            cur_output_rectified = torch.nn.functional.pad(cur_output_rectified, (self.padding[0], self.padding[1] , self.padding[2], self.padding[3]), mode='replicate', value=0)
        
        if flow_smooth != 0:
        
            breetingRoom = flow_force

            for flen in range(0, cur_output_rectified.size(0)):
                
                onlyMax = torch.clamp(torch.max(-mem1[flen,:,:,:], mem2[flen,:,:,:]) - breetingRoom, min=0, max=50)
                onlyMin = -torch.clamp(torch.max(mem1[flen,:,:,:], -mem2[flen,:,:,:]) - breetingRoom, min=0, max=50)

                onlyAll = (onlyMax + onlyMin)

                if self.is_half:
                    onlyAll = onlyAll.float()

                blurInput = np.transpose(cur_output_rectified[flen,:,:,:].numpy(), (2, 1, 0))
                blurFlow = np.transpose(onlyAll.cpu().numpy(), (2, 1, 0))

                
                
                blurred = BlurIt(
                    blurInput, blurFlow, flow_smooth / 10.0
                    )

                cur_output_rectified[flen, :,:,:] = torch.from_numpy(np.transpose(blurred, (2, 1, 0)))

        if alpha:
            cur_output_rectified = torch.cat((cur_output_rectified, alpha_output),dim=1) 
            cur_outputs = [None, cur_output_rectified]
        else:
            cur_outputs = [None,  cur_output_rectified]

        SetTiming("Finish Ending")
        return cur_outputs,None,None

    def displace(self, im, disp_map):
        im = np.asarray(im)
        disp_map = np.asarray(disp_map)
        grid = np.ogrid[list(map(slice, disp_map.shape[:-1]))]
        result = np.zeros_like(im)
        np.add.at(result, tuple((g + disp_map[..., i]) % im.shape[i]
                                for i, g in enumerate(grid)), im)
        return result

    
    def forward_flownets(self, model, input1, input2, time_offsets, flow_force, flow_smooth):

        if time_offsets == None :
            time_offsets = [0.5]
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]
        elif type(time_offsets) == list:
            pass


        flow = [nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(temp)  for temp in [flow_force * model(input1, input2) * 0.5]]
       
        return flow
    

        # Each pair of values represents the number of rows and columns
        # that each element will be displaced

    '''keep this function'''
    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()

        k = 0
        temp = []
        for layers in modulelist:  # self.initScaleNets_offset:
            # print(type(layers).__name__)
            # print(k)
            # if k == 27:
            #     print(k)
            #     pass
            # use the pop-pull logic, looks like a stack.
            if k == 0:
                temp = layers(input)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers,nn.MaxPool2d):
                    stack.push(temp)

                temp = layers(temp)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp,stack.pop()),dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add
            k += 1
        return temp

    '''keep this funtion'''
    def get_MonoNet5(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        # block 7
        model += self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP

        # block 10
        model += self.conv_relu_unpool(32,  16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))

        return  (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

    '''keep this function'''
    @staticmethod
    def FlowProject(inputs, depth = None):
        if depth is not None:
            outputs = [DepthFlowProjectionModule(input.requires_grad)(input,depth) for input in inputs]
        else:
            outputs = [ FlowProjectionModule(input.requires_grad)(input) for input in inputs]
        return outputs



    '''keep this function'''
    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size,
                        padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
        )
        return layers


    '''keep this fucntion'''
    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size,
                        padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=True)
        ])
        return layers

    '''keep this function'''
    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size,
                            padding,kernel_size_pooling):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=True),

            # nn.BatchNorm2d(output_filter),

            nn.MaxPool2d(kernel_size_pooling)
        ])
        return layers

    '''klkeep this function'''
    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size,
                            padding,unpooling_factor):

        layers = nn.Sequential(*[

            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear' , align_corners=True),

            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=True),

            # nn.BatchNorm2d(output_filter),


            # nn.UpsamplingBilinear2d(unpooling_size,scale_factor=unpooling_size[0])
        ])
        return layers
