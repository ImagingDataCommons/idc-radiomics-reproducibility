"""
    ----------------------------------------
    IDC Radiomics use case (Colab)
    
    Functions for data handling/processing
    ----------------------------------------
    
    ----------------------------------------
    Author: Dennis Bontempi
    Email:  dbontempi@bwh.harvard.edu
    ----------------------------------------
    
"""

import os
import json
import numpy as np
import SimpleITK as sitk

import subprocess

## ----------------------------------------

def normalise_volume(input_volume, new_min_val, new_max_val, old_min_val = None, old_max_val = None):

  """
  This function normalizes the volume of an image between
  a new minimum and maximum value, given the old minimum and maximum values.

  Parameters:
    - input_volume: a 3D numpy array representing the image to be normalized.
    - new_min_val: a float representing the new minimum value for the normalized image.
    - new_max_val: a float representing the new maximum value for the normalized image.
    - old_min_val: a float representing the old minimum value of the image.
      If None, the minimum value of the input_volume will be used.
    - old_max_val: a float representing the old maximum value of the image.
      If None, the maximum value of the input_volume will be used.

  Returns:
    - a 3D numpy array representing the normalized image
  """
  
  # make sure the input volume is treated as a float volume
  input_volume = input_volume.astype(dtype = np.float16)

  # if no old_min_val and/or old_max_val are specified, default to the np.min() and np.max() of input_volume
  
  curr_min = np.min(input_volume) if old_min_val == None else old_min_val
  curr_max = np.max(input_volume) if old_max_val == None else old_max_val

  # normalise the values of each voxel between zero and one
  zero_to_one_norm = (input_volume - curr_min)/(curr_max - curr_min)

  # normalise between new_min_val and new_max_val
  return (new_max_val - new_min_val)*zero_to_one_norm + new_min_val

## ----------------------------------------
## ----------------------------------------

def compute_center_of_mass(input_mask):
  
  """
  This function computes the center of mass (CoM) of a binary 3D mask.

  Parameters:
    - input_mask: a 3D numpy array representing the binary mask.

  Returns:
    - a 3D numpy array representing the CoM in (x, y, z) coordinates
  
  """

  # sanity check: the mask should be binary
  assert(len(np.unique(input_mask)) <= 2)
  
  # display a warning if the mask is empty
  if len(np.unique(input_mask)) == 1:
    print('WARNING: DICOM RTSTRUCT is empty.')
    return [-1, -1, -1]
  
  # clip mask values between 0 and 1 (e.g., to cope for masks with max val = 255)
  input_mask = np.clip(input_mask, a_min = 0, a_max = 1)
  
  segmask_4d = np.zeros(input_mask.shape + (4, ))

  # create a triplet of grids that will serve as axis for the next step
  y, x, z = np.meshgrid(np.arange(input_mask.shape[1]), 
                        np.arange(input_mask.shape[0]), 
                        np.arange(input_mask.shape[2]))
  
  # populate
  segmask_4d[..., 0] = x 
  segmask_4d[..., 1] = y 
  segmask_4d[..., 2] = z 
  segmask_4d[..., 3] = input_mask
  
  # keep only the voxels belonging to the mask
  nonzero_voxels = segmask_4d[np.nonzero(segmask_4d[:, :, :, 3])]

  # average the (x, y, z) triplets
  com = np.average(nonzero_voxels[:, :3], axis = 0)
    
  return com
  
## ----------------------------------------
## ----------------------------------------

def get_bbox_dict(coord, seg_mask_shape, bbox_size, z_first = True):


  """
  Computes the bounding box of a segmented mask centered around a coordinate, given a specified box size. 
  
  Parameters:
    - coord: a tuple or list of three integers representing the coordinate at the center of the box.
    - seg_mask_shape: a tuple or list of three integers representing the shape of the segmented mask.
    - bbox_size: a tuple or list of three integers representing the size of the bounding box
      in each direction (coronal, sagittal, and longitudinal).
    - z_first: a boolean indicating whether the longitudinal direction is the first dimension (True)
      or the third dimension (False). Defaults to True.

  Returns:
    - a dictionary with the indices of the first and last voxels in each direction (coronal, sagittal, and longitudinal)
      that define the bounding box, formatted as follows:

      {
        'lon': {'first': 89, 'last': 139}
        'cor': {'first': 241, 'last': 291}, 
        'sag': {'first': 150, 'last': 200}  
      }
  """
  
  assert len(int_center_of_mass) == 3
  assert len(bbox_size) == 3
    
  bbox = dict()

  bbox['cor'] = dict()
  bbox['sag'] = dict()
  bbox['lon'] = dict()

  # take care of axes shuffling 
  cor_idx = 1
  lon_idx = 0 if z_first else 2
  sag_idx = 2 if z_first else 0
  
  # bounding box if no exception is raised
  sag_first = int(int_center_of_mass[sag_idx] - np.round(bbox_size[sag_idx]/2))
  sag_last = int(int_center_of_mass[sag_idx] + np.round(bbox_size[sag_idx]/2)) - 1
  
  cor_first = int(int_center_of_mass[cor_idx] - np.round(bbox_size[cor_idx]/2))
  cor_last = int(int_center_of_mass[cor_idx] + np.round(bbox_size[cor_idx]/2)) - 1
  
  lon_first = int(int_center_of_mass[lon_idx] - np.round(bbox_size[lon_idx]/2))
  lon_last = int(int_center_of_mass[lon_idx] + np.round(bbox_size[lon_idx]/2)) - 1

  # print out exceptions
  if sag_last > seg_mask_shape[sag_idx] - 1 or sag_first < 0:
    print('WARNING: the bounding box size exceeds volume dimensions (sag axis)')
    print('Cropping will be performed ignoring the "bbox_size" parameter')
    
  if cor_last > seg_mask_shape[cor_idx] - 1 or cor_first < 0:
    print('WARNING: the bounding box size exceeds volume dimensions (cor axis)')
    print('Cropping will be performed ignoring the "bbox_size" parameter')
    
  if lon_last > seg_mask_shape[lon_idx] - 1 or lon_first < 0:
    print('WARNING: the bounding box size exceeds volume dimensions (lon axis)')
    print('Cropping will be performed ignoring the "bbox_size" parameter')
    
  # take care of exceptions where bbox is bigger than the actual volume
  sag_first = int(np.max([0, sag_first]))
  sag_last = int(np.min([seg_mask_shape[sag_idx] - 1, sag_last]))
  
  cor_first = int(np.max([0, cor_first]))
  cor_last = int(np.min([seg_mask_shape[cor_idx] - 1, cor_last]))
  
  lon_first = int(np.max([0, lon_first]))
  lon_last = int(np.min([seg_mask_shape[lon_idx] - 1, lon_last]))
  
  # populate the dictionary and return it
  bbox['sag']['first'] = sag_first
  bbox['sag']['last'] = sag_last

  bbox['cor']['first'] = cor_first
  bbox['cor']['last'] = cor_last

  bbox['lon']['first'] = lon_first
  bbox['lon']['last'] = lon_last
    
  return bbox
  
## ----------------------------------------
## ----------------------------------------

def export_res_nrrd_from_dicom(dicom_ct_path, dicom_rt_path, output_dir, pat_id,
                               ct_interpolation = 'linear', output_dtype = "int"):
  
  """
  
  This function exports a resampled CT scan and a resampled RT structure set from a DICOM CT and a DICOM RT structure set.
  
  Parameters:
    - dicom_ct_path: a string representing the path of the DICOM CT folder.
    - dicom_rt_path: a string representing the path of the DICOM RT structure set folder.
    - output_dir: a string representing the path of the output directory.
    - pat_id: a string representing the patient ID.
    - ct_interpolation: a string representing the interpolation method to be used for the CT scan resampling.
    - output_dtype: a string representing the data type of the exported nrrd files.

  Returns:
    - a dictionary containing the paths of the exported nrrd files.

  """
  
  out_log = dict()
  
  # temporary nrrd files path (DICOM to NRRD, no resampling)
  ct_nrrd_path = os.path.join(output_dir, 'tmp_ct_orig.nrrd')
  rt_folder = os.path.join(output_dir, pat_id  + '_whole_ct_rt')
  
  # log the labels of the exported segmasks
  rt_struct_list_path = os.path.join(output_dir, pat_id + '_rt_list.txt')
  
  # convert DICOM CT to NRRD file without resampling
  bash_command = list()
  bash_command += ["plastimatch", "convert"]
  bash_command += ["--input", dicom_ct_path]
  bash_command += ["--output-img", ct_nrrd_path]
                   
  # print progress info
  print("Converting DICOM CT to NRRD using plastimatch... ", end = '')
  out_log['dcm_ct_to_nrrd'] = subprocess.call(bash_command)
  print("Done.")
  
  # convert DICOM RTSTRUCT to NRRD file without resampling
  bash_command = list()
  bash_command += ["plastimatch", "convert"]
  bash_command += ["--input", dicom_rt_path]
  bash_command += ["--referenced-ct", dicom_ct_path]
  bash_command += ["--output-prefix", rt_folder]
  bash_command += ["--prefix-format", 'nrrd']
  bash_command += ["--output-ss-list", rt_struct_list_path]
  
  # print progress info
  print("Converting DICOM RTSTRUCT to NRRD using plastimatch... ", end = '')
  out_log['dcm_rt_to_nrrd'] = subprocess.call(bash_command)
  print("Done.")
  
  # look for the labelmap for GTV
  gtv_rt_file = [f for f in os.listdir(rt_folder) if 'gtv-1' in f.lower()][0]
  rt_nrrd_path = os.path.join(rt_folder, gtv_rt_file)
  
  ## ----------------------------------------
  
  # actual nrrd files path 
  res_ct_nrrd_path = os.path.join(output_dir, pat_id + '_ct_resampled.nrrd')
  res_rt_nrrd_path = os.path.join(output_dir, pat_id + '_rt_resampled.nrrd')
  
  # resample the NRRD CT file to 1mm isotropic
  bash_command = list()
  bash_command += ["plastimatch", "resample"]
  bash_command += ["--input", ct_nrrd_path]
  bash_command += ["--output", res_ct_nrrd_path]
  bash_command += ["--spacing", "1 1 1"]
  bash_command += ["--interpolation", ct_interpolation]
  bash_command += ["--output-type", output_dtype]
  
  # print progress info
  print("\nResampling NRRD CT to 1mm isotropic using plastimatch... ", end = '')
  out_log['dcm_nrrd_ct_resampling'] = subprocess.call(bash_command)
  print("Done.")
  
  # resample the NRRD RTSTRUCT file to 1mm isotropic
  bash_command = list()
  bash_command += ["plastimatch", "resample"]
  bash_command += ["--input", rt_nrrd_path]
  bash_command += ["--output", res_rt_nrrd_path]
  bash_command += ["--spacing", "1 1 1"]
  bash_command += ["--interpolation", "nn"]
    
  # print progress info
  print("Resampling NRRD RTSTRUCT to 1mm isotropic using plastimatch... ", end = '')
  out_log['dcm_nrrd_rt_resampling'] = subprocess.call(bash_command)
  print("Done.")

  # clean up
  print("\nRemoving temporary files (DICOM to NRRD, non-resampled)... ", end = '')
  os.remove(ct_nrrd_path)
  print("Done.")
  
  return out_log


## ----------------------------------------
## ----------------------------------------

def export_com_subvolume(ct_nrrd_path, rt_nrrd_path, output_dir, pat_id, crop_size = (150, 150, 150),
                         z_first = True, rm_orig = False):

  """
  This function exports a subvolume centered on the CoM of the GTV and of the same size as the crop_size parameter.

  Parameters:
    - ct_nrrd_path: a string representing the path to the NRRD CT file.
    - rt_nrrd_path: a string representing the path to the NRRD RTSTRUCT file.
    - output_dir: a string representing the path of the output directory.
    - pat_id: a string representing the patient ID.
    - crop_size: a tuple representing the size of the subvolume to be exported. Defaults to 150x150x150.
    - z_first: a boolean indicating whether the z-axis is the first or the last in the NRRD files. Defaults to True.
    - rm_orig: a boolean indicating whether the original CT and RTSTRUCT files should be removed. Defaults to False.

  Returns:
    - a dictionary containing the log of the operations performed.
  """
  
  # sanity check
  assert(os.path.exists(ct_nrrd_path))
  assert(os.path.exists(rt_nrrd_path))
  
  sitk_seg = sitk.ReadImage(rt_nrrd_path)
  seg = sitk.GetArrayFromImage(sitk_seg)
  
  # output dictionary contains info regarding CoM and the cropping op.s
  out_log = dict()
    
  com = compute_center_of_mass(input_mask = seg)
  com_int = [int(coord) for coord in com]
  
  out_log["com"] = com
  out_log["com_int"] = com_int
  
  # if CoM calculation goes wrong, abort returning the out_log so far
  if sum(com_int) < 0:
    print('WARNING: CoM calculation resulted in an error, aborting... ')
    return out_log
  
  # otherwise go on with the processing and the cropping
  bbox = get_bbox_dict(com_int, seg_mask_shape = seg.shape, bbox_size = crop_size)
  
  # make sure no bounding box exceeds the dimension of the volume
  # (should be taken care of already in the get_bbox_dict() function)
  cor_idx = 1
  lon_idx = 0 if z_first else 2
  sag_idx = 2 if z_first else 0 
  
  # less and not leq at the second term --> the last slice of the volume is seg.shape[...] - 1
  assert(bbox['sag']['first'] >= 0 and bbox['sag']['last'] < seg.shape[sag_idx])
  assert(bbox['cor']['first'] >= 0 and bbox['cor']['last'] < seg.shape[cor_idx])
  assert(bbox['lon']['first'] >= 0 and bbox['lon']['last'] < seg.shape[lon_idx])
  
  
  # cropped nrrd files path
  ct_nrrd_crop_path = os.path.join(output_dir, pat_id + '_ct_res_crop.nrrd')
  rt_nrrd_crop_path = os.path.join(output_dir, pat_id + '_rt_res_crop.nrrd')
  
  if z_first:
    xmin = str(bbox["sag"]["first"]); xmax = str(bbox["sag"]["last"])
    ymin = str(bbox["cor"]["first"]); ymax = str(bbox["cor"]["last"])
    zmin = str(bbox["lon"]["first"]); zmax = str(bbox["lon"]["last"])
  else:
    xmin = str(bbox["lon"]["first"]); xmax = str(bbox["lon"]["last"])
    ymin = str(bbox["cor"]["first"]); ymax = str(bbox["cor"]["last"])
    zmin = str(bbox["sag"]["first"]); zmax = str(bbox["sag"]["last"])
    
  
  # crop the NRRD CT file to the crop_size subvolume
  bash_command = list()
  bash_command += ["plastimatch", "crop"]
  bash_command += ["--input", ct_nrrd_path]
  bash_command += ["--output", ct_nrrd_crop_path]
  bash_command += ["--voxels", "%s %s %s %s %s %s"%(xmin, xmax, ymin, ymax, zmin, zmax)]

  # print progress info
  print("\nCropping the resampled NRRD CT to bbox using plastimatch... ", end = '')
  out_log['dcm_nrrd_ct_cropping'] = subprocess.call(bash_command)
  print("Done.")
  
  
  # crop the NRRD RT file to the crop_size subvolume
  bash_command = list()
  bash_command += ["plastimatch", "crop"]
  bash_command += ["--input", rt_nrrd_path]
  bash_command += ["--output", rt_nrrd_crop_path]
  bash_command += ["--voxels", "%s %s %s %s %s %s"%(xmin, xmax, ymin, ymax, zmin, zmax)]
    
  # print progress info
  print("Cropping the resampled NRRD RTSTRUCT to bbox using plastimatch... ", end = '')
  out_log['dcm_nrrd_rt_cropping'] = subprocess.call(bash_command)
  print("Done.")
  
  # log some useful information about the cropping
  log_file_path = os.path.join(output_dir, pat_id + '_crop_log.json')
  with open(log_file_path, 'w') as json_file:
    json.dump(bbox, json_file, indent = 2)
  
  if rm_orig:
    # clean up
    print("\nRemoving the resampled NRRD files... ", end = '')
    os.remove(ct_nrrd_path)
    os.remove(rt_nrrd_path)
    print("Done.")
  
  return out_log
  

## ----------------------------------------
## ----------------------------------------

def get_input_volume(input_ct_nrrd_path):

  """
  This function prepares the data to be ingested by the model.
  It reads a CT scan nrrd file, normalizes the volume intensity, and crops the volume to a size of 50x50x50.
  Here, the input volume is assumed to be a 150x150x150 volume (see the export_com_subvolume function).

  Parameters:
    - input_ct_nrrd_path: a string representing the file path of the CT scan nrrd file.

  Returns:
    - a numpy array of shape (50,50,50) representing the cropped and normalized volume.
  
  """
  sitk_ct_nrdd = sitk.ReadImage(input_ct_nrrd_path)
  ct_nrdd = sitk.GetArrayFromImage(sitk_ct_nrdd)
      
  # volume intensity normalisation, should follow the same procedure as in the original code:
  # https://github.com/modelhub-ai/deep-prognosis/blob/master/contrib_src/processing.py
  ct_nrdd_norm = normalise_volume(input_volume = ct_nrdd,
                                  new_min_val = 0,
                                  new_max_val = 1,
                                  old_min_val = -1024,
                                  old_max_val = 3071)
    
  ct_nrdd_norm_crop = ct_nrdd_norm[50:100, 50:100, 50:100]
  
  return ct_nrdd_norm_crop

  