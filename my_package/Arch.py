import torch




def MyArch():
	cap = torch.cuda.get_device_capability(0)[0]
	if cap < 10:
		cap = cap * 10

	arch = "arch=compute_{cap},code=compute_{cap}".format(cap = cap)

	return ['-gencode',  arch]
 

def AllArch():
	return [
    #'-gencode', 'arch=compute_30,code=sm_30',
	#'-gencode', 'arch=compute_32,code=sm_32',
	#'-gencode', 'arch=compute_35,code=sm_35',
    #'-gencode', 'arch=compute_37,code=sm_37',
	'-gencode', 'arch=compute_50,code=sm_50',
	'-gencode', 'arch=compute_52,code=sm_52',
	'-gencode', 'arch=compute_53,code=sm_53',
	'-gencode', 'arch=compute_60,code=sm_60',
	'-gencode', 'arch=compute_61,code=sm_61',
	'-gencode', 'arch=compute_62,code=sm_62',
    '-gencode', 'arch=compute_70,code=sm_70',
	'-gencode', 'arch=compute_72,code=sm_72',
	'-gencode', 'arch=compute_75,code=sm_75',
    #'-gencode', 'arch=compute_80,code=sm_80',
    #'-gencode', 'arch=compute_86,code=sm_86',
    #'-gencode', 'arch=compute_86,code=compute_86'
]

def GetArchs():
	getAll = True
	if getAll:
		return AllArch()
	else:
		return MyArch()