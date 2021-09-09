import numpy as np 

def decode_mask(mask):
	pixels = mask.T.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return ' '.join(str(x) for x in runs)

mask = np.array([[0,0,0,0,0,0,0,0,0],[0,0,1,1,1,1,0,0,0],[0,0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0,0],[0,0,0,0,1,1,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,0,0,0],[0,0,0,0,1,1,0,0,0],[0,0,0,0,1,1,0,0,0], [0,0,0,0,0,0,0,0,0]])

decode_mask(mask)
