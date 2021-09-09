import cv2
import numpy as np
from osgeo import gdal
from multiprocessing import Process, JoinableQueue, Lock
from joblib import Parallel, delayed


draw_first_N_img_pair = 0

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
    img_band_cloud_masks = []
    for iBand in range(0, indataset.RasterCount):
        inband = indataset.GetRasterBand(1 + iBand)
        outband_arr=np.array(inband.ReadAsArray())

        ## .max() + .mean() ### .max() + .mean() ### .max() + .mean() ### .max()
        bandmax = outband_arr.max()
        bandmean = outband_arr.mean()
        num_zeros = (outband_arr == 0).sum()
        num_others = (outband_arr > 0).sum()
        bandmean = bandmean*(num_zeros + num_others)/num_others
        outband_arr = (outband_arr*128.0/bandmean)
        outband_arr[outband_arr > 255] = 255

        # ### cv2.equalizeHist ### cv2.equalizeHist ### cv2.equalizeHist 
        # outband_arr = (outband_arr*255.0/bandmax)
        # outband_arr = outband_arr.astype('uint8')
        # outband_arr = cv2.equalizeHist(outband_arr) ### OPENCV

        ### CLOUDS ### CLOUDS ### CLOUDS ### CLOUDS  
        band_cloud_mask = (outband_arr >= 255*1.999)*255
        # band_cloud_mask = cv2.blur((band_cloud_mask > 0)*100.0,(25,25))
        # band_cloud_mask = (band_cloud_mask > 10.0)*255
        img_band_cloud_masks.append(band_cloud_mask)
        # outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_Xband_cloud_mask_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        # outdataset.GetRasterBand(1).WriteArray(band_cloud_mask)

        #outband_arr[band_cloud_mask>0] = 0
        img_bands.append(outband_arr.astype('uint8'))

        if iBand%4>=draw_first_N_img_pair: continue # Draw only N images
        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_Xband_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(outband_arr)
    
    img_cloud_mask = img_band_cloud_masks[0]
    for icloud in range(0, 8):
        img_cloud_mask[img_band_cloud_masks[icloud] > 0] = 255

    grad_cloud_mask = img_bands[0] * 0
    band_grads = []
    for iBand in range(0, indataset.RasterCount):
        ## DX + DY
        dx = cv2.Sobel(img_bands[iBand],cv2.CV_64F,1,0,ksize=9)
        dy = cv2.Sobel(img_bands[iBand],cv2.CV_64F,0,1,ksize=9)

        grad = np.absolute(dx) + np.absolute(dy)
        # grad = grad * (16.0/(9*9)/256)
        grad = grad * 32.0 / (grad.mean())

        #if iBand >=4: grad_cloud_mask[grad > 0.75*grad.max()] = 255

        grad[grad > 255] = 255
        #grad = (grad > gradmean)*255.0

        band_grads.append(grad)

        if iBand%4>=draw_first_N_img_pair: continue # Draw only N images
        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_Xgrad_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(grad)

    band_grad_diff = []
    for iBand in range(0, 4):
        grad_diff = band_grads[iBand + 4].astype('int') - band_grads[iBand].astype('int')
        #grad_diff = grad_diff.astype('uint8')
        #grad_diff = grad_diff * 8.0
        grad_diff[grad_diff > 255] = 255
        grad_diff[grad_diff < 0] = 0
        grad_diff[img_cloud_mask > 0] = 0
        
        ### FORM CORRECT 
        # grad_diff = cv2.medianBlur(grad_diff.astype('uint8'), 5)
        # kernel = np.ones((2,2),np.uint8)
        # grad_diff = cv2.morphologyEx(grad_diff, cv2.MORPH_CLOSE, kernel)

        band_grad_diff.append(grad_diff)

        if iBand%4>=draw_first_N_img_pair: continue # Draw only N images
        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XX_GradDif_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(grad_diff)

    img_band_diff = []
    band_diff_thr = [28, 18, 16, 24]
    for iBand in range(0, 4):
        diff_arr = np.absolute(img_bands[iBand].astype('int') - img_bands[iBand+4].astype('int'))

        diff_arr = diff_arr*255.0 / band_diff_thr[iBand]
        diff_arr[diff_arr > 255] = 255
        diff_arr[img_cloud_mask > 0] = 0

        diff_arr = diff_arr.astype('uint8')

        ### FORM CORRECT 
        diff_arr = cv2.medianBlur(diff_arr, 5)
        # kernel = np.ones((2,2),np.uint8)
        # diff_arr = cv2.morphologyEx(diff_arr, cv2.MORPH_CLOSE, kernel)

        img_band_diff.append(diff_arr)

        if iBand%4>=draw_first_N_img_pair: continue # Draw only N images
        outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XX_ImgDiff_" + BandName[iBand] + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
        outdataset.GetRasterBand(1).WriteArray(diff_arr)

    ### RESULT ### RESULT ### RESULT ### RESULT 
    ### RESULT ### RESULT ### RESULT ### RESULT 
    ### RESULT ### RESULT ### RESULT ### RESULT 
    bound_mask = cv2.blur((img_bands[0] == 0) * 255,(5,5))

    ### COLOR DIFF INTERSECT ### COLOR DIFF INTERSECT ### COLOR DIFF INTERSECT 
    #band_color_dif = np.maximum(np.maximum(img_band_diff[0], img_band_diff[1]), np.maximum(img_band_diff[2], img_band_diff[3]))
    band_color_dif = img_band_diff[0].astype('int') + img_band_diff[1] + img_band_diff[2] + img_band_diff[3]
    band_color_dif = (band_color_dif / 4.0)
    band_color_dif = band_color_dif.astype('uint8')
    #band_color_dif[band_color_dif > 255] = 255
    #band_color_dif = cv2.medianBlur(band_color_dif.astype('uint8'), 5)
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXX_COLORS" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(band_color_dif)

    ### COLORS GARAD ### COLORS GARAD ### COLORS GARAD 
    dx = cv2.Sobel(band_color_dif,cv2.CV_64F,1,0,ksize=9)
    dy = cv2.Sobel(band_color_dif,cv2.CV_64F,0,1,ksize=9)
    grad_colors = np.absolute(dx) + np.absolute(dy)
    grad_colors[bound_mask > 0] = 0
    # grad_colors = grad_colors * 32.0 / (grad_colors.mean())
    # grad_colors[grad_colors > 255] = 255
    grad_colors = grad_colors * 255.0 / (grad_colors.max())
    grad_colors[grad_colors > 255] = 255

    #grad_colors = cv2.Canny(band_color_dif, 300, 500)

    grad_colors = (grad_colors).astype('uint8')
    #band_color_dif = cv2.medianBlur(band_color_dif.astype('uint8'), 5)
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXX_GRADS_COLORS" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(grad_colors)

    ### GRAD INTERSECT ### GRAD INTERSECT ### GRAD INTERSECT 
    band_grad_mask = band_grad_diff[0].astype('int') + band_grad_diff[1] + band_grad_diff[2] + band_grad_diff[3]
    band_grad_mask[bound_mask > 0] = 0
    #band_grad_mask = (band_grad_mask / 4.0)
    band_grad_mask = band_grad_mask * 255.0 / (band_grad_mask.max())
    band_grad_mask[band_grad_mask > 255] = 255
    band_grad_mask = band_grad_mask.astype('uint8')
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXX_GRADS_BAND" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(band_grad_mask)

    ### INTERSECT ALL ### INTERSECT ALL ### INTERSECT ALL ###
    band_grad_mask = np.minimum(band_grad_mask, grad_colors)
    # band_grad_mask = band_grad_mask.astype('int') + grad_colors
    # band_grad_mask = band_grad_mask * 0.5
    band_grad_mask = cv2.medianBlur(band_grad_mask.astype('uint8'), 5)

    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXX_INTERSECT_GRADS" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(band_grad_mask)

    band_grad_mask = (band_grad_mask >= 0.25 * 255)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(77,77))
    band_grad_mask = cv2.morphologyEx(band_grad_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)
    # band_grad_mask = cv2.blur((band_grad_mask > 0)*100.0,(31,31))
    # band_grad_mask = (band_grad_mask > 10.0)*255
    intersect = np.minimum(band_color_dif, band_grad_mask)
    #intersect[intersect > 255] = 255
    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXX_INTERSECT_RES" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(intersect)

    ### THR RESULT ### THR RESULT ### THR RESULT ### THR RESULT 
    intersect = (intersect > 0.66 * 255) * 255
    intersect[bound_mask > 0] = 0

    outdataset = gdal.GetDriverByName('GTiff').Create( outfolder + outfile + "_XXXX_RESULT" + ".tif" , indataset.RasterXSize, indataset.RasterYSize, 1, gdal.GDT_Byte)
    outdataset.GetRasterBand(1).WriteArray(intersect)

    mask = decode_mask((intersect > 0).astype(bool))
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
             #map(delayed(run_change_det), [listdir(infolder)[1]]))

file_submission.close()
print("Done")
