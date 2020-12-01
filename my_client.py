import my_DAIN_class
import clitest
import setting
import torch
import traceback

def CallClient(debug = False):
	data = clitest.Execute()
	try:
		#torch.cuda.set_device(data.sel_process)
		dain_class = my_DAIN_class.DainClass()
		'''
		if debu:
			data.use_half = 0
			data.batch_size = 3
			data.useBenchmark = 0
		'''
		#print("AAAAA")
		#data.inputType = 3
		dain_class.RenderVideo(data)
	except Exception as e:
		tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
		print(str("".join(tb_str)))
		#print(str(e))

if __name__ == "__main__":
	
	if clitest.args.cli:
		CallClient()
	