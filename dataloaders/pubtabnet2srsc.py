# Convert structure information in PubTabNet dataset to start row, end row, start column and end column indexing.
# start row (sr), end row (er), start column (sc), end column (ec) being 0-indexed.
import json
import cv2
import matplotlib.pyplot as plt

# load json file
root = '/Users/i23271/Downloads/table/datasets/PubTabNet/examples/'
annotation_filename = root + 'PubTabNet_Examples.jsonl'

annotations = []
with open(annotation_filename, 'r') as jaf:
    for line in jaf:
        annotations.append(json.loads(line))

# Parsing list of annotations for each of the table
for annot in annotations:
    # Load image to check outcome
    img_filename = root + annot['filename']
    img = cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)
    cell_structure = annot['html']['structure']['tokens']
    cell_annotation = annot['html']['cells']
    # print(cell_structure)
    # print(cell_annotation)
    # import pdb; pdb.set_trace()

    # Divide cell structure into rows to get row-wise division
    # " rowspan="x"" and " colspan="x"" information available only for
    # cell-text data as an attribute for <td> tag. 
    row_div_lst = []
    row_count = 0

    mover = trailer = 0
    while (mover <= len(cell_structure) - 2) and (trailer <= len(cell_structure) - 2):
        if cell_structure[mover] != '</tr>':
            mover += 1
        elif cell_structure[mover] == '</tr>':
            row_div_lst.append(cell_structure[trailer : mover + 1])
            row_count += 1
            mover = trailer = mover + 1
    
    # Sanity checks
    # print(row_div_lst)
    # print("Number of rows: {}".format(row_count))
    # print(mover, trailer, len(cell_structure))
    # td_end_count = 0
    # for i in cell_structure:
    #     if i == '</td>':
    #         td_end_count += 1
    # print(td_end_count, len(cell_annotation))  


    # Identify max column count
    max_col_count = 0
    
    for row_select in row_div_lst:
        tag_id = 0
        col_count = 0
        while tag_id != len(row_select):
            if 'colspan' in row_select[tag_id]:
                span = int(row_select[tag_id].split('=')[-1][1])
                col_count += span
                tag_id += 3
            elif row_select[tag_id] == '</td>':
                col_count += 1
                tag_id += 1
            else:
                tag_id += 1
        
        if col_count > max_col_count:
            max_col_count = col_count
    print("Number of columns: {}".format(max_col_count))

    # Divide rows using "</td>" tag
    # len of cell_annotation i.e. cell_idxs is equal to td_end_count. Use this for appending sr, er etc.
    cell_idx = 0
    for idx, row in enumerate(row_div_lst):
        # start row is the idx, end row is either idx + 1 or idx + rowspan
        mover = trailer = 0
        col_idx = 0
        while (mover <= len(row) - 1) and (trailer <= len(row) - 1):
            if row[mover] != '</td>':
                mover += 1
            elif row[mover] == '</td>':
                cell_block = row[trailer: mover + 1]
                cell_annotation[cell_idx]['start_row'] = idx
                cell_annotation[cell_idx]['start_col'] = col_idx
                
                if cell_block[-2] == '<td>':
                    cell_annotation[cell_idx]['end_row'] = idx + 1
                elif 'rowspan' in cell_block[-3]:
                    rowspan = int(cell_block[-3].split('=')[-1][1])
                    cell_annotation[cell_idx]['end_row'] = idx + rowspan
                else:
                    # might have colspan but no rowspan
                    cell_annotation[cell_idx]['end_row'] = idx + 1

                if cell_block[-2] == '<td>':
                    cell_annotation[cell_idx]['end_col'] = col_idx + 1
                    col_idx += 1
                elif 'colspan' in cell_block[-3]:
                    colspan = int(cell_block[-3].split('=')[-1][1])
                    cell_annotation[cell_idx]['end_col'] = col_idx + colspan
                    col_idx += colspan
                else:
                    # might have rowspan but no colspan
                    cell_annotation[cell_idx]['end_col'] = col_idx + 1

                # Tracking cell_idx to annotate sr, sc, er, and ec information to each cell text
                # cell_idx follows raster fashion.   
                cell_idx += 1
                mover = trailer = mover + 1

    # # Sanity 
    # print(cell_annotation)
    # print("*" * 10)
    # print(cell_structure)
    # for annot in cell_annotation:
    #     if 'start_row' in annot and 'start_col' in annot and 'end_row' in annot and 'end_col' in annot:
    #         continue
    #     else:
    #         print(annot)
    #         print('missing_info')
    #         import pdb; pdb.set_trace()
    # plt.imshow(img)
    # plt.show()
    import pdb; pdb.set_trace()
    
