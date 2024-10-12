import os
import sys
from BookScanSplit import *


input_folder_path = '/Users/reingel/Downloads/jj/'
output_folder_path = '/Users/reingel/Downloads/jj/output/'
debug_folder_path = '/Users/reingel/Downloads/jj/debug/'

# mode = 'text' or 'photo'
bss = BookScanSplit('photo', input_folder_path, output_folder_path, debug_folder_path)
bss.clear_output_folders()
bss.split()

print('All done.')
