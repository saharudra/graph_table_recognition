# Resizing images while preserving aspect ratio
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io
import skimage.transform
from distutils.version import LooseVersion

IMAGE_RESIZE_MODE = "square"
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024
IMAGE_MIN_SCALE = 0

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def cal_adj_label(data_row, data_col, y_row, y_col):
        """
        :input:
            :y_row: adjacency matrix for cells that are part of the same row
            :y_col: adjacency matrix for cells that are part of the same col
        :output:
            :y_adj: adjacency matrix for cells that are immediate neigbhors either
                    row or column wise.
        """
        
        row_edge_index = data_row.edge_index.cpu().numpy()
        col_edge_index = data_col.edge_index.cpu().numpy()

        num_cells = data_row.pos.shape[0]

        adjacency_mat = np.zeros((num_cells, num_cells))

        # Add using rows only
        for row_idx in range(num_cells):
            # Add first same row/col connection directly
            # Check if second cell with same row/col connection
            # is in same col/row as that of previously added 
            # adjacency connections
            row_added = False
            end_cells_added = []
            
            curr_edge_ends = np.where(row_edge_index[0] == row_idx)[0]
            for end in curr_edge_ends:
                # end node of an edge in the edge_index of pytorch geometric data loader output
                end_cell_idx = row_edge_index[1][end]
                # Don't do anything for backward linking edges
                if end_cell_idx < row_idx:
                    continue
                elif y_row[end] == 1 and not(row_added):
                    adjacency_mat[row_idx][end_cell_idx] = 1
                    end_cells_added.append(end_cell_idx)
                    row_added = True
                elif y_row[end] == 1 and row_added:
                    # check if end_cell_idx is in the same col as end_cells_added
                    # for the given row_idx
                    # If in the same col then add to adj_mat else its part
                    # of the same row but not adjacent
                    if check_same_col(end_cell_idx, end_cells_added, y_col, col_edge_index):
                        adjacency_mat[row_idx][end_cell_idx] = 1
                        end_cells_added.append(end_cell_idx)
        
        # Add using cols only
        for col_idx in range(num_cells):
            col_added = False
            end_cells_added = []

            curr_edge_ends = np.where(col_edge_index[0] == col_idx)[0]
            for end in curr_edge_ends:
                end_cell_idx = col_edge_index[1][end]
                if end_cell_idx < col_idx:
                    continue
                elif y_col[end] == 1 and not(col_added):
                    adjacency_mat[col_idx][end_cell_idx] = 1
                    end_cells_added.append(end_cell_idx)
                    col_added = True
                elif y_col[end] == 1 and col_added:
                    if check_same_row(end_cell_idx, end_cells_added, y_row, row_edge_index):
                        adjacency_mat[col_idx][end_cell_idx] = 1
                        end_cells_added.append(end_cell_idx)

        # Copy upper triangular matrix to lower one as adjacency is symmetrical
        adjacency_mat = adjacency_mat + adjacency_mat.T - np.diag(np.diag(adjacency_mat))
        
        return adjacency_mat


def check_same_col(end_cell_idx, end_cells_added, y_col, col_edge_index):
        same_col = True
        for prev_cell in end_cells_added:
            gt_idx = np.intersect1d(np.where(col_edge_index[0] == end_cell_idx), 
                                    np.where(col_edge_index[1] == prev_cell))
            if len(gt_idx) == 0:
                same_col = False
            elif len(gt_idx) == 1 and y_col[gt_idx[0]] == 1:
                continue
            else:
                same_col = False

        return same_col


def check_same_row(end_cell_idx, end_cells_added, y_row, row_edge_index):
    same_row = True
    for prev_cell in end_cells_added:
        gt_idx = np.intersect1d(np.where(row_edge_index[0] == end_cell_idx), 
                                np.where(row_edge_index[1] == prev_cell))
        if len(gt_idx) == 0:
            same_row = False
        elif len(gt_idx) == 1 and y_row[gt_idx[0]] == 1:
            continue
        else:
            same_row = False

    return same_row
