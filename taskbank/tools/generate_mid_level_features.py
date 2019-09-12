# coding: utf-8

# ## Goal
# Process an image to generate curvature maps and vanishing points
#
# ### How?
# - the only argument to the script is the full path to an image.
# - we store the output into the same folder containing the image 
#
# ### Resources
# Using pretrained model available [here](https://github.com/StanfordVL/taskonomy/tree/master/taskbank)

# In[1]:

import sys
import json
import os
import subprocess
import skimage.io
import numpy as np
from subprocess import STDOUT, Popen, PIPE
from pathlib import Path


# #### Inputs
g_task_names = ["curvature", "vanishing_point"]


def compute_top_roi(img_w, img_h):
    if img_w > img_h:
        return (0, 0, img_h, img_h)
    else:
        return (0, 0, img_w, img_w)


def compute_bottom_roi(img_w, img_h):
    if img_w > img_h:
        return (img_w - img_h, 0, img_w, img_h)
    else:
        return (0, img_h - img_w, img_w, img_h)


def crop_image(img, roi):
    return img[roi[1] : roi[3], roi[0] : roi[2], :]


def split_into_square_patches_and_save(full_image_name):
    """
    split an image into two square images for mid-level cue processing
    """
    image_name = os.path.basename(full_image_name)
    top_squared_image_name = os.path.join(
        "/tmp/", image_name.split(".")[0] + "_0." + image_name.split(".")[1]
    )
    bottom_squared_image_name = os.path.join(
        "/tmp", image_name.split(".")[0] + "_1." + image_name.split(".")[1]
    )
    #
    if os.path.exists(full_image_name):
        img = skimage.io.imread(full_image_name, plugin="imageio")
        top_roi = compute_top_roi(img.shape[1], img.shape[0])
        top_img = crop_image(img, top_roi)
        skimage.io.imsave(top_squared_image_name, top_img)
        #
        bottom_roi = compute_bottom_roi(img.shape[1], img.shape[0])
        bottom_img = crop_image(img, bottom_roi)
        skimage.io.imsave(bottom_squared_image_name, bottom_img)
    else:
        print("{} doesn't exist.".format(full_image_name))
    return top_squared_image_name, bottom_squared_image_name


def fuse_curvature_info_in(map_1, map_2):
    """
    We want to keep the maximum of the two response maps while fusing them.
    map_1 is the complete response map and map_2 is the half filled 
    final response map.
    """
    choose_2 = ((map_1 > 0) & (map_1 < map_2)) | ((map_1 < 0) & (map_2 < map_1))
    fused_map = map_1
    fused_map[choose_2] = map_2[choose_2]
    return fused_map


def combine_response_maps(map_0, map_1, img_shape):
    top_roi = compute_top_roi(img_shape[1], img_shape[0])
    bottom_roi = compute_bottom_roi(img_shape[1], img_shape[0])
    #
    aspect_ratio = float(img_shape[1]) / img_shape[0]
    if aspect_ratio < 1:
        final_rows = int(map_0.shape[0] / aspect_ratio)
        final_cols = map_0.shape[1]
    else:
        final_cols = int(map_0.shape[1] * aspect_ratio)
        final_rows = map_0.shape[0]
    #
    print("final rows = {}, final cols = {}".format(final_rows, final_cols))
    final_response_map = np.zeros(
        (final_rows, final_cols, map_0.shape[2]), dtype=map_0.dtype
    )
    final_response_map[: map_0.shape[0], : map_0.shape[1], :] = map_0
    if aspect_ratio > 1:
        col_start = int(map_1.shape[1])
        final_response_map[:, -col_start:, :] = fuse_curvature_info_in(
            map_1, final_response_map[:, -col_start:, :]
        )
    else:
        row_start = int(map_1.shape[0])
        final_response_map[-row_start:, :, :] = fuse_curvature_info_in(
            map_1, final_response_map[-row_start:, :, :]
        )
    return final_response_map



def run_cmd(cmd):
    print(cmd)
    returned_value = subprocess.Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    print("stdout: \n " + returned_value.stdout.read().decode())
    print("stderr: \n " + returned_value.stderr.read().decode())


def construct_prediction_filename(image_name, task_name):
    first_part = image_name.split(".")[0]
    return first_part + "_" + task_name + "_pred.npy"


def find_shape_of(full_image_name):
    if os.path.exists(full_image_name):
        img = skimage.io.imread(full_image_name, plugin="imageio")
        return img.shape
    else:
        return (0, 0, 0)


def compute_curvature(full_image_name):
    # split the image into 2 square patches
    top_cropped_image_name, bottom_cropped_image_name = split_into_square_patches_and_save(
        full_image_name
    )
    cmd_top = "python {} --task curvature --img {} --store {} --store-pred".format(
        script_name,
        top_cropped_image_name,
        top_cropped_image_name.split(".")[0] + "_curvature.png",
    )
    run_cmd(cmd_top)
    cmd_bottom = "python {} --task curvature --img {} --store {} --store-pred".format(
        script_name,
        bottom_cropped_image_name,
        bottom_cropped_image_name.split(".")[0] + "_curvature.png",
    )
    run_cmd(cmd_bottom)
    # read the response maps and combine them
    top_response_map = np.load(
        construct_prediction_filename(top_cropped_image_name, "curvature")
    )
    bottom_response_map = np.load(
        construct_prediction_filename(bottom_cropped_image_name, "curvature")
    )
    # store the combined map into the designated output folder
    full_response_map = combine_response_maps(
        top_response_map, bottom_response_map, find_shape_of(full_image_name)
    )
    print("shape of the final full response map {}".format(full_response_map.shape))
    output_filename = os.path.join(output_folder, os.path.basename(full_image_name))
    np.save(
        construct_prediction_filename(output_filename, "curvature"), full_response_map
    )

def initialize():
    global src_folder, dl_folder, img_folder, output_folder, script_name
    src_folder = "/Users/mishraka/workspace/data/empty_rooms/"
    img_folder = src_folder
    output_folder = "/Users/mishraka/workspace/output/"
    dl_folder = "/Users/mishraka/workspace/taskonomy/taskbank/"
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("correct usage: {} <full_path_image_name>".format(sys.argv[0]))
        sys.exit(1)
    #
    initialize()
    dl_folder = str(Path(__file__).resolve().parents[1])
    print(dl_folder)
    src_folder = str(Path(sys.argv[1]).resolve().parents[0])
    print(src_folder)
    output_folder = src_folder
    img_folder = src_folder
    #
    task_names = g_task_names
    script_name = os.path.join(dl_folder, "tools/run_img_task.py")

    # ####  Process each image now!
    image_name = os.path.basename(sys.argv[1])
    image_filename = os.path.join(img_folder, image_name)
    for task_name in task_names:
        if task_name == "curvature":
            compute_curvature(image_filename)
        else:
            output_image_name = ( image_name.split(".")[0] + "_" + task_name + ".png")
            output_filename = os.path.join(output_folder, output_image_name)
            cmd = "python {} --task {} --img {} --store {} --store-pred".format( script_name, task_name, image_filename, output_filename)
            print(cmd)
            returned_value = subprocess.Popen( cmd, shell=True, stdout=PIPE, stderr=PIPE)
            print("stdout: \n " + returned_value.stdout.read().decode())
            print("stderr: \n " + returned_value.stderr.read().decode())
