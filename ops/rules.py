import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

def naive_gaussian(pos):
    adjacency_matrix = []
    for cell_pos in pos:
        x_cp, y_cp = cell_pos[0].item(), cell_pos[1].item()

        align = {}

        for i, align_pos in enumerate(pos):
            align[i] = {'x': np.exp(-25 * ((x_cp * 1024 - align_pos[0].item() * 1024) / 1024 ) ** 2),
                        'y': np.exp(-25 * ((y_cp * 1024 - align_pos[1].item() * 1024) / 1024 ) ** 2)}
        print(align)
        # Plot alignments
        color = np.array([166, 216, 84]) / 255.0
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        doc_img = img.squeeze(0).permute(1, 2, 0).numpy()
        print(doc_img.shape)
        ax[0].imshow(doc_img, cmap='gray')
        ax[1].imshow(doc_img, cmap='gray')

        # Plot left alignments
        for i, p in enumerate(pos):
            (x, y) = (pos[i][0].item() * 1024, pos[i][1].item() * 1024)
            ax[0].add_patch(pat.Circle((x, y), 10, facecolor=color, edgecolor='k', fill=True, alpha=align[i]['x']))
        
        ax[0].add_patch(pat.Circle((x_cp * 1024, y_cp * 1024), 5, facecolor='r', edgecolor='k', fill=True))
        ax[0].set_title('Column alignment probabilities')

        # Plot left alignments
        for i, p in enumerate(pos):
            (x, y) = (pos[i][0].item() * 1024, pos[i][1].item() * 1024)
            ax[1].add_patch(pat.Circle((x, y), 10, facecolor=color, edgecolor='k', fill=True, alpha=align[i]['y']))
        
        ax[1].add_patch(pat.Circle((x_cp * 1024, y_cp * 1024), 5, facecolor='r', edgecolor='k', fill=True))
        ax[1].set_title('Row alignment Probabilities')
        
        plt.show()

        import pdb; pdb.set_trace()