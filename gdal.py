def decode_mask(mask):
	pixels = mask.T.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return ' '.join(str(x) for x in runs)

import numpy as np

from osgeo import gdal
gdal.TermProgress = gdal.TermProgress_nocb

infolder = "f:\@Data\@Satellite\!!! roscosmos-rucode\Images_composit\Images_composit\8_ch/"
infile = "KVI_20180922_SCN4_UN93__KV1_20190804_SCN2_UN94.tif"
indataset = gdal.Open(infolder + infile, gdal.GA_ReadOnly )

outfolder = "f:\@Data\@Satellite\!!! roscosmos-rucode\Images_out/"
outfile = "OUT__" + infile

for iBand in range(1, indataset.RasterCount + 1):
    inband = indataset.GetRasterBand(iBand)
    print( outfolder + str(iBand) + "_" + outfile )
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + str(iBand) + "_" + outfile , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outband = outdataset.GetRasterBand(1)

    for i in range(inband.YSize - 1, -1, -1):
        scanline = inband.ReadAsArray(0, i, inband.XSize, 1, inband.XSize, 1)
        scanline = np.choose( np.equal( scanline, None),
                                       (scanline, None) )
        outband.WriteArray(scanline, 0, i)

        indataset.GetRasterBand(iBand)
    
    outband_arr=np.array(outband.ReadAsArray())
    mask = decode_mask((outband_arr > 0).astype(bool))
    print(mask)
    print("\n\n\n\n\n")


### OUT ### OUT ### OUT ### OUT ### OUT ### OUT ### OUT 0
### OUT ### OUT ### OUT ### OUT ### OUT ### OUT ### OUT 0
### OUT ### OUT ### OUT ### OUT ### OUT ### OUT ### OUT 0
