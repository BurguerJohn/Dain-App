counter = 0
interpolations = None
padding = None

def AddCounter(tag):
	global counter
	#print("\nSetting:\n" +tag +": " + str(counter) + "\n")
	counter += 1
def SetPad(pad):
	global padding
	padding = pad
def GetPad():
	global padding
	return padding

