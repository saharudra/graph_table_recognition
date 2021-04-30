"""
Task Description:
    Preprocessing script to convert ICDAR 2013 into SciTSR dataset format.

Take the reg and str information from the ICDAR 2013 dataset and convert it into SciSR structure, chunk and rel information.

Chunk:
    Stores the bounding box information in raster fashion. Bboxes are provided as (x1, y1, x2, y2) of the top-left and bottom-right corners.

Structure:
    Stores cell id, cell text, cell content as list of individual words, start-row, end-row, start-column, end-column of each of the cells

    See e.g. below:
    {   
        "id": 68,
        "tex": "DPFL (2-stream)~",
        "content": [
            "DPFL",
            "(2-stream)"
        ],
        "start_row": 14,
        "end_row": 14,
        "start_col": 0,
        "end_col": 0
	}

    rows and cols are 0-indexed.

Img:
    Close crop of the table images

Pdf:
    pdf page of the table without processing
"""
import os
import cv2
import numpy as np
import json
import operator

from ops.misc import mkdir_p

import xml.etree.ElementTree as ET
from pdf2image import convert_from_path
from PyPDF2 import PdfFileWriter, PdfFileReader

data_folder = '/data/rudra/table_structure_recognition/datasets/icdar_2013_comp/competition-dataset-eu'
dst_folder = '/data/rudra/table_structure_recognition/datasets/icdar/2013_eu'
chunk_dst = dst_folder + '/chunk'
mkdir_p(chunk_dst)
structure_dst = dst_folder + '/structure'
mkdir_p(structure_dst)
img_dst = dst_folder + '/img'
mkdir_p(img_dst)
pdf_dst = dst_folder + '/pdf'
mkdir_p(pdf_dst)
rel_dst = dst_folder + '/rel'
mkdir_p(rel_dst)

filelist = os.listdir(data_folder)

unique_file_prefix = []

for f in filelist:
    if f[0] == '.':
        continue
    else:
        f = '-'.join(f.split('.')[0].split('-')[:2])
        unique_file_prefix.append(f)

unique_file_prefix = list(set(unique_file_prefix))
print(unique_file_prefix)
# import pdb; pdb.set_trace()
# Create bounding box matrix and content list
# length of content_list == length of bbox_matrix
bbox_matrix = []
content_list = []

"""
str.xml file consists of bounding box information as x1, y1, x2, y2 of bottom-left and top-right corner.
First convert this to top-left and bottom-right corner information before adding to bbox_matrix.
Also convert the direction of the y-axis as it runs from bottom to top in str.xml file of ICDAR 2013 dataset
whereas from top to bottom for SciTSR dataset.

Read pdf at 72 dpi to identify the image size for chunk creation

Chunk file: dictionary of "chunks" with list of them in raster fashion

For structure, if end_row or end_col is not provided then end_row = start_row + row-increment, end_col = start_col + col-increment
Maintain id based on cells being captured.

STEP 1:
    Get table images and table pdfs
    Get chunks for all of the tables
    Get preliminary structures for tables
"""
img_count = 0
for f in unique_file_prefix:
    # try:
    str_xml_file = data_folder + "/" + f + "-str.xml"
    reg_xml_file = data_folder + "/" + f + "-reg.xml"
    pdf_file = data_folder + "/" + f + ".pdf"

    images = convert_from_path(pdf_file, dpi=72)  # Placing under try except as not all files had their corresponding pdfs.
    images = [np.array(img) for img in images]  # List of pdf images
    img_0 = images[0]
    img_size = img_0.shape
    x_len, y_len = img_size[1], img_size[0]  # pdf file as an image, every file is of same length
    pdfs = PdfFileReader(open(pdf_file, 'rb'))

    xml_root = ET.parse(str_xml_file).getroot()
    xml_root_reg = ET.parse(reg_xml_file).getroot()

    # Get table region and page number from xml_root_reg
    table_regions = []
    table_page_number = []
    for idx, table in enumerate(xml_root_reg):
        for region in table:
            table_page_number.append(int(region.attrib['page']) - 1)  # Making it 0 indexed

            for tags in region:
                if tags.tag == 'bounding-box':
                    bbox_dict = tags.attrib
                    x1_bl = tags.attrib['x1']
                    y1_bl = tags.attrib['y1']
                    x2_tr = tags.attrib['x2']
                    y2_tr = tags.attrib['y2']

                    x1_tl = int(tags.attrib['x1'])
                    x2_br = int(tags.attrib['x2'])
                    y1_tl = int(y_len - int(tags.attrib['y2']))
                    y2_br = int(y_len - int(tags.attrib['y1']))

                    table_regions.append([x1_tl, y1_tl, x2_br, y2_br])
    for idx, table in enumerate(xml_root):
        # concatenating using '-' to make it easier to split
        chunk_file = chunk_dst + '/' + f + "-" + table.attrib['id'] + ".chunk"   # <file_name-table_id.chunk>
        chunk_dict = {"chunks": []}

        struct_file = structure_dst + "/" + f + '-' + table.attrib['id'] + ".json"  # <file_name-table_id.json>
        struct_dict = {"cells": []}

        img_file = img_dst + "/" + f + '-' + table.attrib['id'] + ".png"  # <file_name-table_id.png>
        print(img_file)
        pdf_file = pdf_dst + "/" + f + '-' + table.attrib['id'] + ".pdf"  # <file_name-table_id.pdf>

        # Get table image
        curr_img = images[table_page_number[idx]]  # get page with table
        curr_img_bbox = table_regions[idx]
        x1, y1, x2, y2 = curr_img_bbox
        curr_table_img = curr_img[y1:y2, x1:x2]
        cv2.imwrite(img_file, cv2.cvtColor(curr_table_img, cv2.COLOR_BGR2RGB))

        # Get table pdf
        curr_pdf = pdfs.getPage(table_page_number[idx])
        curr_pdf_out = PdfFileWriter()
        curr_pdf_out.addPage(curr_pdf)
        # import pdb; pdb.set_trace()
        with open(pdf_file, 'wb') as pdfStream:
            curr_pdf_out.write(pdfStream)
        
        for region in table:
            # ON CURRENT REGION
            col_increment = max(0, int(region.attrib['col-increment']))  # Found -1 row-increment in us-018-str.xml for icdar2013
            row_increment = max(0, int(region.attrib['row-increment']))
            img_page_num = int(region.attrib['page']) - 1  # Making it 0-indexed

            for idx, cell in enumerate(region):
                # ON CURRENT CELL
                curr_chunk_dict = {}
                curr_struct_dict = {}
                curr_chunk_dict["id"] = idx  # 0 indexed id for the task, Adding ID SciTSR doesn't have it in chunk files
                curr_struct_dict["id"] = idx  # 0 indexed id for the task

                for tags in cell:
                    # Already in raster fashion
                    if tags.tag == 'content':
                        # content_list.append(tags.text)
                        curr_chunk_dict['text'] = tags.text
                        curr_struct_dict['tex'] = tags.text
                        curr_struct_dict['content'] = tags.text.split()

                    elif tags.tag == 'bounding-box':
                        curr_bbox_dict = tags.attrib
                        x1_bl = tags.attrib['x1']
                        y1_bl = tags.attrib['y1']
                        x2_tr = tags.attrib['x2']
                        y2_tr = tags.attrib['y2']

                        x1_tl = float(tags.attrib['x1'])
                        x2_br = float(tags.attrib['x2'])
                        y1_tl = float(int(tags.attrib['y2']))
                        y2_br = float(int(tags.attrib['y1']))
                        curr_chunk_dict['pos'] = [x1_tl, x2_br, y1_tl, y2_br]
                        # Add top left and bottom right information to bbox matrix
                        # bbox_matrix.append([x1_tl, y1_tl, x2_br, y2_br])
                    
                start_col = int(cell.attrib['start-col'])
                start_row = int(cell.attrib['start-row'])
                if 'end-row' in cell.attrib:
                    end_row = int(cell.attrib['end-row'])
                else:
                    end_row = start_row + row_increment
                
                if 'end-col' in cell.attrib:
                    end_col = int(cell.attrib['end-col'])
                else:
                    end_col = start_col + col_increment
                
                curr_struct_dict["start_row"] = start_row
                curr_struct_dict["start_col"] = start_col
                curr_struct_dict["end_row"] = end_row
                curr_struct_dict["end_col"] = end_col

                chunk_dict['chunks'].append(curr_chunk_dict)
                struct_dict['cells'].append(curr_struct_dict)

        print(chunk_dict)
        print(struct_dict)
        with open(chunk_file, 'w+') as jcf:
            json.dump(chunk_dict, jcf, indent=2)

        with open(struct_file, 'w+') as jsf:
            json.dump(struct_dict, jsf, indent=2)
    # except Exception as e:
    #     print('sh')
    #     print(e)
    #     print('shhh')
    #     # import pdb; pdb.set_trace()
    #     continue


"""
STEP 1b:
    Read chunks and structs back and convert them to raster fashion
"""
chunk_filelist = os.listdir(chunk_dst)

for f in chunk_filelist:
    curr_chunk_file = os.path.join(chunk_dst, f)
    f_name, f_ext = os.path.splitext(f)
    curr_cell_file = os.path.join(structure_dst, f_name + '.json')

    with open(curr_chunk_file, 'rb') as cf, open(curr_cell_file, 'rb') as sf:
        chunks_data = json.load(cf)
        structs_data = json.load(sf)

    chunks_data = chunks_data['chunks']
    it = len(chunks_data)
    structs_data = structs_data['cells']

    table_matrix = []

    for i in range(it):
        chunk = chunks_data[i]
        cell = structs_data[i]
        x1, x2, y1, y2 = chunk['pos']
        cell_start_row, cell_start_col = cell['start_row'], cell['start_col']
        cell_info = [x1, y1, x2, y2, cell_start_row, cell_start_col, chunk, cell]
        table_matrix.append(cell_info)
    
    table_matrix = sorted(table_matrix, key=operator.itemgetter(4, 5))

    chunks_dict = {}
    chunks_dict['chunks'] = []
    structs_dict = {}
    structs_dict['cells'] = []

    for idx, cell in enumerate(table_matrix):
        curr_chunk = cell[-2]
        curr_struct = cell[-1]
        curr_chunk["id"] = idx
        curr_struct["id"] = idx
        chunks_dict['chunks'].append(curr_chunk)
        structs_dict['cells'].append(curr_struct)
    
    with open(curr_chunk_file, 'w') as cf, open(curr_cell_file, 'w') as sf:
        json.dump(chunks_dict, cf, indent=2)
        json.dump(structs_dict, sf, indent=2)


"""
STEP 2:
    Use preliminary structures to add fundamental empty cells to structures
    Use chunks and structures to build rel

Read files from structure_dst and find the fun_table_matrix for the current table
"""

struct_filelist = os.listdir(structure_dst)

for f in struct_filelist:
    curr_filename = structure_dst + '/' + f
    with open(curr_filename, 'rb') as csf:
        curr_strcut_dict = json.load(csf)
    
    curr_max_row = 0
    curr_max_col = 0
    max_id = 0

    for cell in curr_strcut_dict['cells']:
        if cell['end_row'] > curr_max_row:
            curr_max_row = cell['end_row']
        if cell['end_col'] > curr_max_col:
            curr_max_col = cell['end_col']
        if cell['id'] > max_id:
            max_id = cell['id']
    
    new_id = max_id + 1
    fun_table_matrix = np.zeros((curr_max_row, curr_max_col))

    for cell in curr_strcut_dict['cells']:
        cell_start_row = cell['start_row']
        cell_start_col = cell['start_col']
        cell_end_row = cell['end_row']
        cell_end_col = cell['end_col']

        fun_table_matrix[cell_start_row : cell_end_row + 1, cell_start_col : cell_end_col + 1] = 1 

    rows, cols = fun_table_matrix.shape[0], fun_table_matrix.shape[1]

    for row in range(rows):
        for col in range(cols):
            if fun_table_matrix[row][col] == 0.0:
                new_cell = {}
                new_cell['id'] = new_id
                new_id += 1

                new_cell['tex'] = ""
                new_cell['content'] = []
                new_cell['start_row'] = row
                new_cell['start_col'] = col 
                new_cell['end_row'] = row
                new_cell['end_col'] = col

                curr_strcut_dict['cells'].append(new_cell)

    with open(curr_filename, 'w') as csf:
        json.dump(curr_strcut_dict, csf, indent=2)

"""
STEP 3:
    Read chunk files and structure files to create relation file
    Relation file captures horizontal and vertical relationship between cell ids from chunk file 
    ID_1  ID_2  REL:Blanks
        REL = 1 for Horizontal, 2 for Vertical
        Blanks = Number of blank 
    Read a chunk from chunk file, find it's start and end rows and columns from structure file for ID_1
        The idx of the chunk and the cell should be same as it was stored in raster fashion before fundamental empty cells were added to structure file.
        Validate this with matching cell['tex'] and chunk['text']
        Otherwise perform single pass through cells
    Perform the same for ID_2
    Relation:
        Horizontal: rows overlap such that one is equal to or absorbs the other, end_col of ID_1 is 1 less than start_col of ID_2.
        Vertical: cols overlap such that one is equal to or absorbs the other, end_row of ID_1 is 1 less than start_row of ID_2
        Thus relationship is captured as vertically down and towards the right.
"""

chunks_filelist = os.listdir(chunk_dst)

def overlap(min1, max1, min2, max2):
    if min2 <= min1 <= max2 or min1 <= min2 <= max1:
      return True
    else:
      return False


for f in chunks_filelist:
    chunk_file = os.path.join(chunk_dst, f)
    f_name, f_ext = os.path.splitext(f)
    struct_file = os.path.join(structure_dst, f_name + ".json")

    with open(chunk_file, 'r') as cf, open(struct_file, 'r') as sf:
        chunks_data = json.load(cf)
        structs_data = json.load(sf)

    rel_file = rel_dst + '/' + f_name + '.rel'

    rel_list = []  # Each relation is a list as: [ID_1, ID_2, Rel, Blanks]

    chunks_data = chunks_data['chunks']
    structs_data = structs_data['cells']

    it = len(chunks_data)
    for i in range(it):
        chunk_data_i = chunks_data[i]
        strcut_data_i = structs_data[i]
        start_row_i = strcut_data_i['start_row']
        end_row_i = strcut_data_i['end_row']
        start_col_i = strcut_data_i['start_col']
        end_col_i = strcut_data_i['end_col']

        for j in range(i+1, it):
            chunk_data_j = chunks_data[j]
            struct_data_j = structs_data[j]
            start_row_j = struct_data_j['start_row']
            end_row_j = struct_data_j['end_row']
            start_col_j = struct_data_j['start_col']
            end_col_j = struct_data_j['end_col']

            # Relationship can be either vertical or horizontal or no_rel
            # Check for horizontal relationship
            # If i and j are horizontally related, i overlaps j and start of j happens after end of i
            # Similarly for vertical relationship
            if len(rel_list) > 0:
                prev_rel = rel_list[-1] 
            
            if overlap(start_row_i, end_row_i, start_row_j, end_row_j) and end_col_i < end_col_j:
                if (len(rel_list) == 0):
                    horizontal_rel = [i, j, 1, start_col_j - end_col_i - 1, start_col_i, end_col_i, start_col_j, end_col_j]
                # Not previously present, cannot be present and already have a vertical relationship
                elif i != prev_rel[0]:
                    horizontal_rel = [i, j, 1, start_col_j - end_col_i - 1, start_col_i, end_col_i, start_col_j, end_col_j]
                # If already added, see if the new horizontal cell overlaps vertically with previous horizontally related cell
                elif i == prev_rel[0] and overlap(prev_rel[-2], prev_rel[-1], start_col_j, end_col_j):
                    horizontal_rel = [i, j, 1, start_col_j - end_col_i - 1, start_col_i, end_col_i, start_col_j, end_col_j]
                rel_list.append(horizontal_rel)

            if overlap(start_col_i, end_col_i, start_col_j, end_col_j) and end_row_i < end_row_j:
                if len(rel_list) == 0:
                    vertical_rel = [i, j, 2, start_row_j - end_row_i - 1, start_row_i, end_row_i, start_row_j, end_row_j]
                # No previsouly present or present but only had horizontal relationship
                elif (i != prev_rel[0]) or (i == prev_rel[0] and prev_rel[2] == 1):
                    vertical_rel = [i, j, 2, start_row_j - end_row_i - 1, start_row_i, end_row_i, start_row_j, end_row_j]
                # If already added, see if the new vertical cell overlaps horizontally with the previous vertically related cell
                elif i == prev_rel[0] and prev_rel[2] == 2 and overlap(prev_rel[-2], prev_rel[-1], start_row_j, end_row_j):
                    vertical_rel = [i, j, 2, start_row_j - end_row_i - 1, start_row_i, end_row_i, start_row_j, end_row_j]
                rel_list.append(vertical_rel)

    # Take unique
    rel_set = set([tuple(t) for t in rel_list])
    rel_list = sorted([list(t) for t in list(rel_set)], key=operator.itemgetter(0, 1))

    # Write relations to rel file
    with open(rel_file, 'w+') as rel_file:
        for rel in rel_list:
            rel_file.write(str(rel[0]) + '\t' + str(rel[1]) + '\t' + str(rel[2]) + ':' + str(rel[3]))
            rel_file.write('\n')

# import pdb; pdb.set_trace()
