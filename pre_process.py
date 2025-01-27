import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# Path to your NIfTI file
file_path = 'BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii'
# Load the NIfTI file
img = nib.load(file_path)

# Get the image data (in NumPy array form)
img_data = img.get_fdata()

# Check the shape of the data (this should give you an idea of the dimensions)
print("Image shape:", img_data.shape)

TRAIN_DATASET_PATH = r'BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

test_image_flair=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_flair.nii').get_fdata() # Added a forward slash / to the path
print(test_image_flair.max())

test_image_flair=scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)
test_image_t1=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_t1.nii').get_fdata()
test_image_t1=scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1ce=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_t1ce.nii').get_fdata()
test_image_t1ce=scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

test_image_t2=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_t2.nii').get_fdata()
test_image_t2=scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

test_mask=nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_seg.nii').get_fdata()
test_mask=test_mask.astype(np.uint8)

print(np.unique(test_mask))  #0, 1, 2, 4 (Need to reencode to 0, 1, 2, 3)
test_mask[test_mask==4] = 3  #Reassign mask values 4 to 3
print(np.unique(test_mask))

t2_list = sorted(glob.glob(r'BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_list = sorted(glob.glob(r'BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
flair_list = sorted(glob.glob(r'BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask_list = sorted(glob.glob(r'BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

# Ensure necessary imports
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler

# Assuming scaler is already defined in a previous cell
scaler = MinMaxScaler()

# Assuming t2_list, t1ce_list, flair_list, and mask_list are already defined in previous cells
for img in range(len(t2_list)):   # Using t1_list as all lists are of same size
	print("Now preparing image and masks number: ", img)

	temp_image_t2 = nib.load(t2_list[img]).get_fdata()
	temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

	temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
	temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

	temp_image_flair = nib.load(flair_list[img]).get_fdata()
	temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

	temp_mask = nib.load(mask_list[img]).get_fdata()
	temp_mask = temp_mask.astype(np.uint8)
	temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3

	temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

	# Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
	# cropping x, y, and z
	temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
	temp_mask = temp_mask[56:184, 56:184, 13:141]

	val, counts = np.unique(temp_mask, return_counts=True)

	if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
		print("Save Me")
		temp_mask = to_categorical(temp_mask, num_classes=4)
		np.save(r'BraTS2020_TrainingData/input_data_3channels/images/image_' + str(img) + '.npy', temp_combined_images)
		np.save(r'BraTS2020_TrainingData/input_data_3channels/masks/mask_' + str(img) + '.npy', temp_mask)
	else:
		print("I am useless")
import splitfolders  # or import split_folders

input_folder = r'BraTS2020_TrainingData\input_data_3channels'
output_folder = r'BraTS2020_TrainingData\input_data_128'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)