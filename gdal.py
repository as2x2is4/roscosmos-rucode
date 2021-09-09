import cv2
import numpy as np
from osgeo import gdal
from multiprocessing import Process, JoinableQueue, Lock
from joblib import Parallel, delayed



def decode_mask(mask):
	pixels = mask.T.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return " ".join(str(x) for x in runs)

def run_change_det(infile):
    BandName = ["1Red0", "2Grn0", "3Blu0", "4NIR0","1Red1", "2Grn1", "3Blu1", "4NIR1",]
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

        img_bands.append(outband_arr.astype('uint8'))

        if iBand%4 != 0: continue # Draw onky red
        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_Xband_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(outband_arr)

    band_grad_bin = []
    for iBand in range(0, indataset.RasterCount):
        #grad = cv2.Laplacian(img_bands[iBand], cv2.CV_64F, ksize=11)
        #dx, dy = cv2.spatialGradient(img_bands[iBand], ksize=3)

        dx = cv2.Sobel(img_bands[iBand],cv2.CV_64F,1,0,ksize=5)
        dy = cv2.Sobel(img_bands[iBand],cv2.CV_64F,0,1,ksize=5)

        grad = np.absolute(dx) + np.absolute(dy)

        num_zeros = (grad == 0).sum()
        num_others = (grad > 0).sum()
        gradmean = grad.mean()*(num_zeros + num_others)/num_others
        grad = (grad*255.0/(gradmean*2.0))
        grad[grad > 255] = 255
        #grad = (grad > gradmean)*255.0

        band_grad_bin.append(grad)

        if iBand%4 != 0: continue # Draw onky red
        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_Xgrad_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(grad)

    band_grad_diff = []
    for iBand in range(0, 4):
        grad_diff = np.absolute(band_grad_bin[iBand + 4].astype('int') - band_grad_bin[iBand].astype('int'))
        #grad_diff = grad_diff.astype('uint8')
        grad_diff[grad_diff < 0] = 0
        band_grad_diff.append(grad_diff)

        if iBand%4 != 0: continue # Draw onky red
        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXgrDif_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(grad_diff)

    img_band_mask = []
    band_diff_thr = [28, 18, 16, 24]
    for iBand in range(0, 4):
        diff_arr = np.absolute(img_bands[iBand].astype('int') - img_bands[iBand+4].astype('int'))
        #bands_cloud_mask = np.absolute(img_bands_cloud[iBand] + img_bands_cloud[iBand+4])

        # Threshold # Threshold # Threshold 
        diff_arr = ( (diff_arr > band_diff_thr[iBand]) * 1 ).astype('uint8')

        # # ### Correct CLOUDS by Threshold
        # thr_cloud  = 0.9
        # # thr_shadow  = 0.4
        # # diff_arr[img_bands[iBand] > img_bands[iBand].max()*thr_cloud] = 0
        # diff_arr[img_bands[iBand+4] > img_bands[iBand+4].max()*thr_cloud] = 0
        # # diff_arr[img_bands[iBand] < img_bands[iBand].max()*thr_shadow] = 0
        # # diff_arr[img_bands[iBand+4] < img_bands[iBand+4].max()*thr_shadow] = 0
        
        ### FORM CORRECT 
        diff_arr = cv2.medianBlur(diff_arr, 3)
        #diff_arr = cv2.bilateralFilter(diff_arr,9,75,75)
        # kernel = np.ones((2,2),np.uint8)
        # diff_arr = cv2.morphologyEx(diff_arr, cv2.MORPH_CLOSE, kernel)

        img_band_mask.append(diff_arr); continue

        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXdiff_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(diff_arr*255)

        #find all your connected components (white blobs in your image)
        #find all your connected components (white blobs in your image)
        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(diff_arr, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        max_size = 9995  
        min_size = 100  

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

    ### RESULT ### RESULT ### RESULT ### RESULT 
    ### RESULT ### RESULT ### RESULT ### RESULT 
    ### RESULT ### RESULT ### RESULT ### RESULT 

    ### GRAD INTERSECT
    band_grad_mask = band_grad_diff[0] + band_grad_diff[1] + band_grad_diff[2] + band_grad_diff[3]
    band_grad_mask = band_grad_mask * 0.5
    band_grad_mask[band_grad_mask > 255] = 255
    band_grad_mask = cv2.medianBlur(band_grad_mask.astype('uint8'), 5)
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXXgradMAsk" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(band_grad_mask)

    ### COLOR DIFF intersect ### COLOR DIFF intersect ### COLOR DIFF intersect 
    intersect = img_band_mask[0] + img_band_mask[1] + img_band_mask[2] + img_band_mask[3]
    intersect = cv2.medianBlur(intersect.astype('uint8'), 5)
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXX_COLORintersect" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(intersect*64.0)

    ### THRESHOLD
    intersect = intersect*64.0 + band_grad_mask*0.5
    intersect[intersect > 255] = 255
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXX_THRESHOLD" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(intersect)

    ###return

    ### intersect + bound_mask ### intersect + bound_mask ### intersect + bound_mask 
    intersect = (intersect > 196)*1
    bound_mask = cv2.blur((img_bands[0] == 0) * 255,(5,5))
    intersect[bound_mask > 0] = 0

    # cv2.morphologyEx
    kernel = np.ones((3,3),np.uint8)
    intersect = cv2.morphologyEx(intersect, cv2.MORPH_CLOSE, kernel)

    ### connectedComponentsWithStats ### connectedComponentsWithStats 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(intersect, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    max_size = 9995
    min_size = 200

    #your answer image
    cc_filter = np.zeros((intersect.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            if sizes[i] <= max_size:
                cc_filter[output == i + 1] = 1

    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXXresult" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(cc_filter*255)

    mask = decode_mask((cc_filter > 0).astype(bool))
    file_submission.write(infile.replace('.tif', '') + "," + mask + '\n')
    print(infile + " -- Done")



######### FUN END ### FUN END ### FUN END ### FUN END ### FUN END 
######### FUN END ### FUN END ### FUN END ### FUN END ### FUN END 
######### FUN END ### FUN END ### FUN END ### FUN END ### FUN END 


infolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_composit/Images_composit/8_ch/"
#infolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/mask/img/"

outfolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_out_sub/"
#outfolder = "f:/@Data/@Satellite/!!! roscosmos-rucode/Images_out/"

file_submission = open(outfolder + "submission.csv", "w")
file_submission.write("Id,mask" + '\n')

from os import listdir
# for infile in listdir(infolder):#[3:4]:
#     run_change_det(infile)

Parallel(n_jobs=-1, verbose=0, backend="threading")(
             map(delayed(run_change_det), listdir(infolder)))

file_submission.close()
print("Done")
