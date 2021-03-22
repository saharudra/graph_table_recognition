from PIL import Image
import pytesseract
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.image as img
import numpy as np
import pandas as pd


def get_text(file, visualize=False):
    # Process image with Tesseract
    document = Image.open(file)
    df = pytesseract.image_to_data(document, lang='eng', output_type=pytesseract.Output.DICT)

    # Visualize document image and bounding boxes
    if visualize:
        doc_img = img.imread(file)
        color = np.array([166, 216, 84]) / 255.0

        fig, ax = plt.subplots(1)
        ax.imshow(doc_img, cmap='gray')
        for i, word in enumerate(df['text']):
            if int(df['conf'][i]) > 60:
                (x, y, w, h) = (df['left'][i], df['top'][i], df['width'][i], df['height'][i])
                ax.add_patch(pat.Rectangle((x, y), w, h, color=color, fill=True, alpha=0.3, edgecolor=color))
                ax.text(x + int(w/2) - 3, y + 3, df['text'][i], fontsize=7)

    print(' ')
    return df, document._size


# Compute and visualize alignment probabilities
def alignment_probabilities(df, dims):
    # Get the cell corresponding to the word "den"
    print(df)
    idx = df['text'].index('ISPA')
    cell = dict([(field, df[field][idx]) for field in ['left', 'top', 'width', 'height', 'text']])
    align = {}

    for i, word in enumerate(df['text']):
        # Can skip the NaNs
        if int(df['conf'][i]) > 60.0:
            align[i] = {'left': np.exp(-25*((cell['left'] - df['left'][i]) / dims[0])**2),
                        'top': np.exp(-25*((cell['top'] - df['top'][i]) / dims[1])**2)}

    # Plot the alignments
    color = np.array([166, 216, 84]) / 255.0
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    doc_img = img.imread('/Users/i23271/Downloads/table/datasets/SciTSR/test/img/9912012v1.4.png')
    ax[0].imshow(doc_img, cmap='gray')
    ax[1].imshow(doc_img, cmap='gray')

    # Plot left alignments
    for i, word in enumerate(df['text']):
        if int(df['conf'][i]) > 60.0:
            (x, y, w, h) = (df['left'][i], df['top'][i], df['width'][i], df['height'][i])
            ax[0].add_patch(pat.Rectangle((x, y), w, h, facecolor=color, edgecolor='k',
                                          fill=True, alpha=align[i]['left']))
            display = '{0}, {1:3.2f}'.format(df['text'][i], align[i]['left'])
            ax[0].text(x + int(w / 2), y + 4, display, fontsize=7)
    (x, y, w, h) = (cell['left'], cell['top'], cell['width'], cell['height'])
    ax[0].add_patch(pat.Rectangle((x, y), w, h, facecolor='r', edgecolor='k', fill=True))
    ax[0].set_title('Left alignment probabilities')

    # Plot top alignments
    for i, word in enumerate(df['text']):
        if int(df['conf'][i]) > 60.0:
            (x, y, w, h) = (df['left'][i], df['top'][i], df['width'][i], df['height'][i])
            ax[1].add_patch(pat.Rectangle((x, y), w, h, facecolor=color, edgecolor='k',
                                          fill=True, alpha=align[i]['top']))
            display = '{0}, {1:3.2f}'.format(df['text'][i], align[i]['left'])
            ax[1].text(x + int(w / 2), y + 4, display, fontsize=7)
    (x, y, w, h) = (cell['left'], cell['top'], cell['width'], cell['height'])
    ax[1].add_patch(pat.Rectangle((x, y), w, h, facecolor='r', edgecolor='k', fill=True))
    ax[1].set_title('Top alignment probabilities')
    plt.show()
    import pdb; pdb.set_trace()
    return None


if __name__ == '__main__':
    plt.ion()
    # text, doc_dims = get_text('./samples/test-european.jpg', visualize=True)
    text, doc_dims = get_text('/Users/i23271/Downloads/table/datasets/SciTSR/test/img/9912012v1.4.png', visualize=True)
    alignment_probabilities(text, doc_dims)
