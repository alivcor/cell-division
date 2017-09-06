import scipy.io as sio
import numpy as np
import cv2
import h5py
import os
from scipy import misc



def augment_image(image, mask):
    return_images = np.zeros((8,image.shape[0], image.shape[1]))
    return_masks = np.zeros((8,mask.shape[0], mask.shape[1]))

    image_flip = cv2.flip(image, 1)

    return_images[0, :, :] = image
    return_images[1, :, :] = np.rot90(image)
    return_images[2, :, :] = np.rot90(return_images[1, :, :])
    return_images[3, :, :] = np.rot90(return_images[2, :, :])

    return_images[4, :, :] = image_flip
    return_images[5, :, :] = np.rot90(return_images[4, :, :])
    return_images[6, :, :] = np.rot90(return_images[5, :, :])
    return_images[7, :, :] = np.rot90(return_images[7, :, :])

    # same for mask
    return_masks[0, :, :] = mask
    return_masks[1, :, :] = np.rot90(mask)
    return_masks[2, :, :] = np.rot90(return_masks[1, :, :])
    return_masks[3, :, :] = np.rot90(return_masks[2, :, :])

    return_masks[4, :, :] = image_flip
    return_masks[5, :, :] = np.rot90(return_masks[4, :, :])
    return_masks[6, :, :] = np.rot90(return_masks[5, :, :])
    return_masks[7, :, :] = np.rot90(return_masks[7, :, :])

    return return_images, return_masks




def selective_crop(frame_of_interest=0, mask_image, num_patches=150, cell_ratio_threshold=0.15, strain_file_path="/Users/abhinandandubey/Desktop/Mesh/523-FULLDATA/20161129 - Strain 201 (Chl1 deletion) - Position 0.mat", ):
    # num_patches = 150
    # cell_ratio_threshold = 0.15

    print("Starting to crop selectively..")

    # for now lets load current_image as one of the frames from Strain201 and current_mask from a random mask
    # frame_of_interest = 0
    # mask_image = cv2.imread('mask-12.png')  # multiply 255 after doing croppings

    # strain_file_path = "/Users/abhinandandubey/Desktop/Mesh/523-FULLDATA/20161129 - Strain 201 (Chl1 deletion) - Position 0.mat"
    dataset = h5py.File(strain_file_path, 'r')
    assert dataset["cDIC"].shape[0] == dataset["c488"].shape[0] == dataset["c561"].shape[0]
    print "dataset[\"cDIC\"].shape : " + str(dataset["cDIC"].shape)

    corresponding_c488 = np.mean(dataset["c488"][frame_of_interest, :, :, :], axis=0)  # Mean Pooling
    corresponding_c561 = np.mean(dataset["c561"][frame_of_interest, :, :, :], axis=0)  # Mean Pooling

    print "corresponding_c488.shape : " + str(corresponding_c488.shape)
    print "corresponding_c561.shape : " + str(corresponding_c561.shape)

    original_image = dataset["cDIC"][frame_of_interest, 1, :, :]  # Take only first channel for original image
    mask_image = mask_image[:, :, 2]  # Do same for original image
    print "original_image.shape : " + str(original_image.shape)
    print "mask_image.shape : " + str(mask_image.shape)

    # Now we're done loading stuff, lets come to selective cropping

    dataset = np.zeros((num_patches * 8, 224, 224, 3))
    masks = np.zeros((num_patches * 8, 224, 224))
    coordinates = np.zeros((num_patches * 8, 3))

    print "dataset.shape : " + str(dataset.shape)
    print "masks.shape : " + str(masks.shape)
    print "coordinates.shape : " + str(coordinates.shape)

    patch_begin = 1
    patch_end = 448
    success_count = 0

    while (success_count < num_patches - 1):
        locs = np.random.randint(patch_begin, high=patch_end, size=(2, 2))
        cropped_mask = mask_image[locs[1, 0]:locs[1, 0] + 64, locs[0, 0]:locs[0, 0] + 64]
        # print "locs[1,0]:64, locs[0,0]:64 : " + str(locs[1,0]) + ":64, " + str(locs[0,0]) + ":64"
        # print "cropped_mask.shape : " + str(cropped_mask.shape)
        # print cropped_mask
        # print "max : " + str(np.amax(cropped_mask))
        cell_ratio = np.sum(np.sum(cropped_mask)) / (64 * 64)
        # print "cell_ratio : " + str(cell_ratio)
        if (cell_ratio > cell_ratio_threshold):
            print "\n\n----------------------------------------------------------------------------------------------"
            print "SC: " + str(success_count + 1)
            print "Voila ! cell_ratio = " + str(cell_ratio)
            success_count = success_count + 1
            print "Updated success count, cropping now..."
            cropped_image = original_image[locs[1, 0]:locs[1, 0] + 64, locs[0, 0]:locs[0, 0] + 64]
            cropped_c488 = corresponding_c488[locs[1, 0]:locs[1, 0] + 64, locs[0, 0]:locs[0, 0] + 64]
            cropped_c561 = corresponding_c561[locs[1, 0]:locs[1, 0] + 64, locs[0, 0]:locs[0, 0] + 64]
            print "cropped_image.shape : " + str(cropped_image.shape)
            print "cropped_c488.shape : " + str(cropped_c488.shape)
            print "cropped_c561.shape : " + str(cropped_c561.shape)
            print "Resizing.."
            # resize the guys
            cropped_image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_NEAREST)
            cropped_c488 = cv2.resize(cropped_c488, (224, 224), interpolation=cv2.INTER_NEAREST)
            cropped_c561 = cv2.resize(cropped_c561, (224, 224), interpolation=cv2.INTER_NEAREST)
            cropped_mask = cv2.resize(cropped_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            print "cropped_image.shape : " + str(cropped_image.shape)
            print "cropped_c488.shape : " + str(cropped_c488.shape)
            print "cropped_c561.shape : " + str(cropped_c561.shape)
            print "cropped_mask.shape : " + str(cropped_mask.shape)

            augmented_images, augmented_masks = augment_image(image=cropped_image, mask=cropped_mask)
            print "augmented_images.shape : " + str(augmented_images.shape)
            print "augmented_masks.shape : " + str(augmented_masks.shape)

            augmented_c488, augmented_c561 = augment_image(cropped_c488, cropped_c561)
            print "augmented_c488.shape : " + str(augmented_c488.shape)
            print "augmented_c561.shape : " + str(augmented_c561.shape)

            print "dataset.shape : " + str(dataset.shape)
            dataset[success_count * 8: (success_count + 1) * 8, :, :, 0] = augmented_images
            dataset[success_count * 8: (success_count + 1) * 8, :, :, 1] = augmented_c488
            dataset[success_count * 8: (success_count + 1) * 8, :, :, 2] = augmented_c561
            print "dataset.shape : " + str(dataset.shape)

    return dataset


