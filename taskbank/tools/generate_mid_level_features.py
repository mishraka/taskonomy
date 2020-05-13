# coding: utf-8

# ## Goal
# find mid level visual cues of an image or the images in a given folder
#
# ### Resources
# Using pretrained model available [here](https://github.com/StanfordVL/taskonomy/tree/master/taskbank)

# In[1]:

import json
import sys
import os
import subprocess
import skimage.io
import numpy as np
import argparse
import image_to_numpy
from subprocess import STDOUT, Popen, PIPE

#  Inputs
g_task_names = ["curvature", "rgb2sfnorm", "vanishing_point", "class_places"]


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
    return img[roi[1]: roi[3], roi[0]: roi[2], :]


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
    img = image_to_numpy.load_image_file(full_image_name)
    print("image shape = ", img.shape)
    #
    if not os.path.exists(top_squared_image_name):
        top_roi = compute_top_roi(img.shape[1], img.shape[0])
        top_img = crop_image(img, top_roi)
        skimage.io.imsave(top_squared_image_name, top_img)
    #
    if not os.path.exists(bottom_squared_image_name):
        bottom_roi = compute_bottom_roi(img.shape[1], img.shape[0])
        bottom_img = crop_image(img, bottom_roi)
        skimage.io.imsave(bottom_squared_image_name, bottom_img)
    return top_squared_image_name, bottom_squared_image_name, img.shape


def combine_response_maps(map_0, map_1, img_shape):
    aspect_ratio = float(img_shape[1]) / img_shape[0]
    if aspect_ratio < 1:
        final_rows = int(map_0.shape[0] / aspect_ratio)
        final_cols = map_0.shape[1]
    else:
        final_cols = int(map_0.shape[1] * aspect_ratio)
        final_rows = map_0.shape[0]
    #
    print("final rows = {}, final cols = {}".format(final_rows, final_cols))
    #
    if map_0.ndim == 2:
        final_response_map = np.zeros(
            (final_rows, final_cols), dtype=map_0.dtype)
        final_response_map[: map_0.shape[0], : map_0.shape[1]] = map_0
        if aspect_ratio > 1:
            col_start = int(map_1.shape[1])
            final_response_map[:, -col_start:] = map_1
        else:
            row_start = int(map_1.shape[0])
            final_response_map[-row_start:, :] = map_1
    elif map_0.ndim == 3:
        final_response_map = np.zeros(
            (final_rows, final_cols, map_0.shape[2]), dtype=map_0.dtype
        )
        #
        final_response_map[: map_0.shape[0], : map_0.shape[1], :] = map_0
        if aspect_ratio > 1:
            col_start = int(map_1.shape[1])
            final_response_map[:, -col_start:, :] = map_1
        else:
            row_start = int(map_1.shape[0])
            final_response_map[-row_start:, :, :] = map_1
    return final_response_map


def run_cmd(cmd):
    print(cmd)
    returned_value = subprocess.Popen(
        cmd, shell=True, stdout=PIPE, stderr=PIPE)
    print("stdout: \n " + returned_value.stdout.read().decode())
    print("stderr: \n " + returned_value.stderr.read().decode())


def construct_prediction_filename(image_name, task_name):
    first_part = image_name.split(".")[0]
    return first_part + "_" + task_name + "_pred.npy"


def find_shape_of(full_image_name):
    if os.path.exists(full_image_name):
        img = skimage.io.imread(full_image_name)
        return img.shape
    else:
        return (0, 0, 0)


def compute_fullimage(full_image_name, taskname):
    # split the image into 2 square patches
    top_cropped_image_name, bottom_cropped_image_name, img_shape = split_into_square_patches_and_save(
        full_image_name
    )
    cmd_top = "python {} --task {} --img {} --store {} --store-pred".format(
        script_name,
        taskname,
        top_cropped_image_name,
        top_cropped_image_name.split(".")[0] + "_{}.png".format(taskname),
    )
    run_cmd(cmd_top)
    cmd_bottom = "python {} --task {} --img {} --store {} --store-pred".format(
        script_name,
        taskname,
        bottom_cropped_image_name,
        bottom_cropped_image_name.split(".")[0] + "_{}.png".format(taskname),
    )
    run_cmd(cmd_bottom)
    # read the response maps and combine them
    top_response_map = np.load(
        construct_prediction_filename(top_cropped_image_name, taskname)
    )
    bottom_response_map = np.load(
        construct_prediction_filename(bottom_cropped_image_name, taskname)
    )
    # store the combined map into the designated output folder
    full_response_map = combine_response_maps(
        top_response_map, bottom_response_map, img_shape
    )
    #
    print("shape of the final full response map {}".format(full_response_map.shape))
    output_filename = os.path.join(
        output_folder, os.path.basename(full_image_name))
    np.save(construct_prediction_filename(
        output_filename, taskname), full_response_map)


def process_image(image_filename):
    for task_name in g_task_names:
        if (
            task_name == "curvature"
            or task_name == "rgb2sfnorm"
            or task_name == "rgb2depth"
        ):
            compute_fullimage(image_filename, task_name)
        else:
            output_image_name = (
                os.path.basename(image_filename).split(".")[0]
                + "_"
                + task_name
                + ".png"
            )
            output_filename = os.path.join(output_folder, output_image_name)

            cmd = "python {} --task {} --img {} --store {} --store-pred".format(
                script_name, task_name, image_filename, output_filename
            )
            run_cmd(cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_arg", help="specify full path of an image or a directory with images"
    )
    parser.add_argument(
        "-output_dir", action="store", dest="output_folder", default="/tmp"
    )
    args = parser.parse_args()
    return args


def find_images_in(dirname):
    all_files = os.listdir(dirname)
    all_files = filter(lambda x: x.endswith(
        (".png", ".jpeg", ".jpg")), all_files)
    return [os.path.join(dirname, x) for x in list(all_files)]


def initialize(results):
    global img_folder, output_folder, script_name
    output_folder = results.output_folder
    dl_folder = os.path.abspath(os.path.dirname(__file__))
    script_name = os.path.join(dl_folder, "run_img_task.py")
    print("script_name = {}, \n output_folder = {}".format(
        script_name, output_folder))


if __name__ == "__main__":
    args = parse_args()
    initialize(args)

    imgs_to_process = []
    if os.path.isfile(args.input_arg):
        img_folder = os.path.dirname(args.input_arg)
        imgs_to_process.append(args.input_arg)
        print("image directory = {}".format(img_folder))
    elif os.path.isdir(args.input_arg):
        img_folder = args.input_arg
        print("{} is a directory".format(args.input_arg))
        imgs_to_process = find_images_in(img_folder)

    print("image to process", imgs_to_process)
    for full_image_name in imgs_to_process:
        print("processing visual cues in {}".format(full_image_name))
        process_image(full_image_name)
