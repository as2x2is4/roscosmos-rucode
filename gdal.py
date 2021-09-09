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

#infolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_composit/Images_composit/8_ch/"
infolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/mask/img/"

#outfolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_out_sub/"
outfolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_out/"

BandName = ["1Red0", "2Grn0", "3Blu0", "4NIR0","1Red1", "2Grn1", "3Blu1", "4NIR1",]

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

        # ### cv2.equalizeHist ### cv2.equalizeHist ### cv2.equalizeHist 
        # outband_arr = ( outband_arr * 255.0 / outband_arr.max() ).astype('uint8')
        # outband_arr = cv2.equalizeHist(outband_arr) ### OPENCV

        ## .max() + .mean() ### .max() + .mean() ### .max() + .mean() ### .max()
        bandmax = outband_arr.max()
        num_zeros = (outband_arr == 0).sum()
        num_others = (outband_arr > 0).sum()
        bandmean = outband_arr.mean()*(num_zeros + num_others)/num_others

        outband_arr = (outband_arr*255.0/(bandmean*2.0))
        outband_arr[outband_arr > 255] = 255

        img_bands.append(outband_arr)

        # outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_BNDeq_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        # outdataset.GetRasterBand(1).WriteArray(outband_arr)

    img_band_mask = []
    for iBand in range(0, 4):
        diff_arr = np.absolute(img_bands[iBand].astype('int') - img_bands[iBand+4].astype('int'))
        #bands_cloud_mask = np.absolute(img_bands_cloud[iBand] + img_bands_cloud[iBand+4])

        # Threshold # Threshold # Threshold 
        diff_arr = ( (diff_arr > 24) * 1 ).astype('uint8')

        # # ### Correct CLOUDS by Threshold
        # thr_cloud  = 0.7
        # thr_shadow  = 0.4
        # diff_arr[img_bands[iBand] > img_bands[iBand].max()*thr_cloud] = 0
        # diff_arr[img_bands[iBand+4] > img_bands[iBand+4].max()*thr_cloud] = 0
        # diff_arr[img_bands[iBand] < img_bands[iBand].max()*thr_shadow] = 0
        # diff_arr[img_bands[iBand+4] < img_bands[iBand+4].max()*thr_shadow] = 0
        
        ### FORM CORRECT 
        diff_arr = cv2.medianBlur(diff_arr, 3)
        #diff_arr = cv2.bilateralFilter(diff_arr,9,75,75)
        # kernel = np.ones((2,2),np.uint8)
        # diff_arr = cv2.morphologyEx(diff_arr, cv2.MORPH_CLOSE, kernel)

        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXdifClose_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(diff_arr*255)

###        img_band_mask.append(diff_arr); continue

        #find all your connected components (white blobs in your image)
        #find all your connected components (white blobs in your image)
        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(diff_arr, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        max_size = 9995  
        min_size = 30  

        #your answer image
        diff_cc_filter = np.zeros((diff_arr.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                if sizes[i] <= max_size:
                    diff_cc_filter[output == i + 1] = 1

        img_dif_thr_mask = diff_cc_filter
        img_band_mask.append(img_dif_thr_mask)

        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXdifCC_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(diff_cc_filter*255)

    intersect = img_band_mask[0] + img_band_mask[1] + img_band_mask[2] + img_band_mask[3]
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXresult" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(((intersect >= 2))*63)

    mask = decode_mask((intersect >= 2).astype(bool))
    file_submission.write(infile.replace('.tif', '') + "," + mask + '\n')

file_submission.close()
print("Done")


### OUT ### OUT ### OUT ### OUT ### OUT ### OUT ### OUT 0
### OUT ### OUT ### OUT ### OUT ### OUT ### OUT ### OUT 0
### OUT ### OUT ### OUT ### OUT ### OUT ### OUT ### OUT 0
