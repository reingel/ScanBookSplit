import os
import sys
from draw_text_regions import *


input_folder_path = '/Users/reingel/Downloads/code-book/input/'
output_folder_path = '/Users/reingel/Downloads/code-book/output/'
debug_folder_path = '/Users/reingel/Downloads/code-book/debug/'

s2p = Scan2Pdf(input_folder_path, output_folder_path, debug_folder_path)
s2p.clear_output_folder()
s2p.clear_debug_folder()
s2p.convert()

print('All done.')
