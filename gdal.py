# https://www.kaggle.com/c/roscosmos-rucode/data

def decode_mask(mask):
	pixels = mask.T.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return " ".join(str(x) for x in runs)

import cv2
import numpy as np

from osgeo import gdal
#gdal.TermProgress = gdal.TermProgress_nocb

infolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_composit/Images_composit/8_ch/"
#infolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/mask/img/"
#outfolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_out/"
outfolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_out_sub/"
BandName = ["Red0", "Grn0", "Blu0", "NIR0","Red1", "Grn1", "Blu1", "NIR1",]


file_submission = open(outfolder + "submission.csv", "w")
file_submission.write("Id,mask" + '\n')

from os import listdir
for infile in listdir(infolder):
    #infile = "KVI_20180922_SCN4_UN93__KV1_20190804_SCN2_UN94.tif"
    outfile = infile

    indataset = gdal.Open(infolder + infile, gdal.GA_ReadOnly )

    img_bands = []
    for iBand in range(0, indataset.RasterCount):
        inband = indataset.GetRasterBand(1 + iBand)
        outband_arr=np.array(inband.ReadAsArray())

        #outband_arr = (outband_arr > 0).astype('uint8')
        outband_arr = np.right_shift(outband_arr, 4).astype('uint8')
        outband_arr = cv2.equalizeHist(outband_arr) ### OPENCV

        img_bands.append(outband_arr)

        # outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        # outdataset.GetRasterBand(1).WriteArray(outband_arr)

    img_band_diffs = []
    img_band_mask = []
    for iBand in range(0, 4):
        #diff_arr = 128 + img_bands[iBand]/2 - img_bands[iBand+4]/2
        diff_arr = np.absolute(img_bands[iBand].astype('int') - img_bands[iBand+4].astype('int'))
        # outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XDiff_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        # outdataset.GetRasterBand(1).WriteArray(diff_arr)
        #diff_arr[diff_arr < 128] = 0
        img_band_diffs.append(diff_arr)

        img_dif_thr_mask = ( (diff_arr > 128) * 1 ).astype('uint8')
        #img_dif_thr_mask = cv2.bilateralFilter(img_dif_thr_mask,9,75,75)
        img_dif_thr_mask = cv2.medianBlur(img_dif_thr_mask, 5)

        img_band_mask.append(img_dif_thr_mask)
        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXThr_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(img_dif_thr_mask*255)

    intersect = img_band_mask[0] + img_band_mask[1] + img_band_mask[2] + img_band_mask[3]
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXres_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(intersect*63)

    mask = decode_mask((intersect >= 3).astype(bool))
    file_submission.write(infile.replace('.tif', '') + "," + mask + '\n')

file_submission.close()

   
    