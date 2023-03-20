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

def get_bbox_dict(coord, volume_shape, bbox_size, z_first=True):

  """
  Computes the bounding box of a segmented mask centered around a coordinate, given a specified box size. 
  
  Parameters:
    - coord: a tuple or list of three integers representing the coordinate at the center of the box.
    - volume_shape: a tuple or list of three integers representing the shape of the volume.
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
  
  if coord.dtype != int:
    coord.astype(int)
  assert len(coord) == 3
  assert len(bbox_size) == 3
    
  bbox = dict()

  bbox['cor'] = dict()
  bbox['sag'] = dict()
  bbox['lon'] = dict()

  # if axes are mirrored 
  cor_idx = 1
  lon_idx = 0 if z_first else 2
  sag_idx = 2 if z_first else 0
  
  # bounding box if no exception is raised
  sag_first = int(coord[sag_idx] - np.round(bbox_size[sag_idx]/2))
  sag_last = int(coord[sag_idx] + np.round(bbox_size[sag_idx]/2)) - 1
  
  cor_first = int(coord[cor_idx] - np.round(bbox_size[cor_idx]/2))
  cor_last = int(coord[cor_idx] + np.round(bbox_size[cor_idx]/2)) - 1
  
  lon_first = int(coord[lon_idx] - np.round(bbox_size[lon_idx]/2))
  lon_last = int(coord[lon_idx] + np.round(bbox_size[lon_idx]/2)) - 1

  # print warning if the bounding box size exceeds the volume dimensions
  # this can happen, e.g., when the tumour is located very close to one of
  # the borders of the CT image (for instance, close to the lung-chest wall interface)
  if sag_last > volume_shape[sag_idx] - 1 or sag_first < 0:
    print('WARNING: the bounding box size exceeds volume dimensions (sag axis)')
    print('Cropping will be performed ignoring the "bbox_size" parameter')
    
  if cor_last > volume_shape[cor_idx] - 1 or cor_first < 0:
    print('WARNING: the bounding box size exceeds volume dimensions (cor axis)')
    print('Cropping will be performed ignoring the "bbox_size" parameter')
    
  if lon_last > volume_shape[lon_idx] - 1 or lon_first < 0:
    print('WARNING: the bounding box size exceeds volume dimensions (lon axis)')
    print('Cropping will be performed ignoring the "bbox_size" parameter')
    
  # take care of the aforementioned exceptions
  sag_first = int(np.max([0, sag_first]))
  sag_last = int(np.min([volume_shape[sag_idx] - 1, sag_last]))
  
  cor_first = int(np.max([0, cor_first]))
  cor_last = int(np.min([volume_shape[cor_idx] - 1, cor_last]))
  
  lon_first = int(np.max([0, lon_first]))
  lon_last = int(np.min([volume_shape[lon_idx] - 1, lon_last]))
  
  # populate the output dictionary
  bbox['sag']['first'] = sag_first
  bbox['sag']['last'] = sag_last

  bbox['cor']['first'] = cor_first
  bbox['cor']['last'] = cor_last

  bbox['lon']['first'] = lon_first
  bbox['lon']['last'] = lon_last
    
  return bbox

## ----------------------------------------
## ----------------------------------------

def crop_around_coord(path_to_volume, path_to_output, coord,
                      crop_size=(150, 150, 150), z_first=True):

  """
  This function exports a crop of the specified volume and size, centered around a coordinate.

  Parameters:
    - path_to_volume: a string representing the path to the volume to crop.
    - path_to_output: a string representing the path of the output file.
    - coord: a tuple representing the coordinates around which to crop.
    - crop_size: a tuple representing the size of the subvolume to be exported. Defaults to 150x150x150.
    - z_first: a boolean indicating whether the z-axis is the first or the last in the NRRD files. Defaults to True.
  """
  
  # sanity check
  assert(os.path.exists(path_to_volume))
  
  sitk_vol = sitk.ReadImage(path_to_volume)
  vol = sitk.GetArrayFromImage(sitk_vol)
      
  # otherwise go on with the processing and the cropping
  bbox = get_bbox_dict(com_int, volume_shape=vol.shape, bbox_size=crop_size)
  
  # make sure no bounding box exceeds the dimension of the volume
  # (should be taken care of already in the get_bbox_dict() function)
  cor_idx = 1
  lon_idx = 0 if z_first else 2
  sag_idx = 2 if z_first else 0 
  
  # sanity check on dimensions
  assert(bbox['sag']['first'] >= 0 and bbox['sag']['last'] < vol.shape[sag_idx])
  assert(bbox['cor']['first'] >= 0 and bbox['cor']['last'] < vol.shape[cor_idx])
  assert(bbox['lon']['first'] >= 0 and bbox['lon']['last'] < vol.shape[lon_idx])
  
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
  bash_command += ["--input", path_to_volume]
  bash_command += ["--output", path_to_output]
  bash_command += ["--voxels", "%s %s %s %s %s %s"%(xmin, xmax, ymin, ymax, zmin, zmax)]

  _ = subprocess.run(bash_command, capture_output=True, check=True)

  # print progress info
  print("\nCropping the volume around the specified coordinates using plastimatch... ", end = '')
  print("Done.")

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

  