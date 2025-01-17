from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda, Add
from keras.optimizers import Adam
from keras.metrics import MeanIoU
import tensorflow as tf

kernel_initializer =  'he_uniform' #Try others if you want
def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'npy'):

            image = np.load(img_dir+image_name)
            image=image.astype(np.float32)
            images.append(image)
    images = np.array(images)

    return(images)
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    #keras needs the generator infinite, so we will use while true
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


def recurrent_block_3d(input_tensor, filters, kernel_size=3, recur_num=2):
    """
    Create a recurrent convolutional block for 3D data
    """
    conv = Conv3D(filters, kernel_size, padding='same', kernel_initializer='he_uniform')(input_tensor)
    conv = BatchNormalization()(conv)
    
    # Recurrent connections
    for _ in range(recur_num):
        conv_res = Conv3D(filters, kernel_size, padding='same', kernel_initializer='he_uniform')(conv)
        conv_res = BatchNormalization()(conv_res)
        conv_res = Add()([conv_res, conv])  # Residual connection
        conv = conv_res
    
    return conv

def residual_recurrent_block_3d(input_tensor, filters, kernel_size=3, recur_num=2):
    """
    Create a residual recurrent block with skip connection
    """
    # Shortcut connection
    shortcut = Conv3D(filters, 1, padding='same', kernel_initializer='he_uniform')(input_tensor)
    
    # Recurrent block
    conv = recurrent_block_3d(input_tensor, filters, kernel_size, recur_num)
    conv = recurrent_block_3d(conv, filters, kernel_size, recur_num)
    
    # Add skip connection
    output = Add()([conv, shortcut])
    return output

def r2_unet_3d(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes, recur_num=2):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = residual_recurrent_block_3d(s, 16, recur_num=recur_num)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    p1 = Dropout(0.1)(p1)

    c2 = residual_recurrent_block_3d(p1, 32, recur_num=recur_num)
    p2 = MaxPooling3D((2, 2, 2))(c2)
    p2 = Dropout(0.1)(p2)

    c3 = residual_recurrent_block_3d(p2, 64, recur_num=recur_num)
    p3 = MaxPooling3D((2, 2, 2))(c3)
    p3 = Dropout(0.2)(p3)

    c4 = residual_recurrent_block_3d(p3, 128, recur_num=recur_num)
    p4 = MaxPooling3D((2, 2, 2))(c4)
    p4 = Dropout(0.2)(p4)

    # Bridge
    c5 = residual_recurrent_block_3d(p4, 256, recur_num=recur_num)
    c5 = Dropout(0.3)(c5)

    # Expansion path
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])  # Skip connection 1
    c6 = residual_recurrent_block_3d(u6, 128, recur_num=recur_num)
    c6 = Dropout(0.2)(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])  # Skip connection 2
    c7 = residual_recurrent_block_3d(u7, 64, recur_num=recur_num)
    c7 = Dropout(0.2)(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])  # Skip connection 3
    c8 = residual_recurrent_block_3d(u8, 32, recur_num=recur_num)
    c8 = Dropout(0.1)(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])  # Skip connection 4
    c9 = residual_recurrent_block_3d(u9, 16, recur_num=recur_num)
    c9 = Dropout(0.1)(c9)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
    return model
""" 
# Test the model
if __name__ == "__main__":
    model = r2_unet_3d(128, 128, 128, 3, 4, recur_num=2)
    print(model.input_shape)
    print(model.output_shape)
"""
import os
import numpy as np
from keras.optimizers import Adam
import segmentation_models_3D as sm
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf

# Define the file paths and batch size
train_img_dir = r"BraTS2020_TrainingData\input_data_128\train\images\\"
train_mask_dir = r"BraTS2020_TrainingData\input_data_128\train\masks\\"

val_img_dir = r"BraTS2020_TrainingData\input_data_128\val\images\\"
val_mask_dir = r"BraTS2020_TrainingData\input_data_128\val\masks\\"

train_img_list = sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

val_img_list = sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))
batch_size = 2

# Load the data generator
train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list,
                              val_mask_dir, val_mask_list, batch_size)

# Define weights for each class in the loss function
wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25

# Define the loss and metrics
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

# Learning rate and optimizer
LR = 0.0001
optim = Adam(LR)

# Steps per epoch
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

# Initialize and compile the R2U-Net++ model
model = r2_unet_3d(IMG_HEIGHT=128,
                   IMG_WIDTH=128,
                   IMG_DEPTH=128,
                   IMG_CHANNELS=3,
                   num_classes=4,
                   recur_num=2)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
""" 
# Define callbacks
# Save the best model based on validation IoU score
checkpoint = ModelCheckpoint('r2unet2_plus_3d_best_model.hdf5', 
                             monitor='val_iou_score', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

# Save training and validation metrics to a CSV file
csv_logger = CSVLogger('r2unet2_plus_training_log.csv', append=True)
"""


# Train the model
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=10,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch)

# Save the final model
model.save('r2unet2_plus_3d_final_model.hdf5')

# Save training history as a dictionary
np.save('r2unet2_plus_training_history.npy', history.history)
