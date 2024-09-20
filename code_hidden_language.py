import os
import sys
from BookScanSplit import *


input_folder_path = '/Users/reingel/Downloads/code-book/input/'
output_folder_path = '/Users/reingel/Downloads/code-book/output/'
debug_folder_path = '/Users/reingel/Downloads/code-book/debug/'

bss = BookScanSplit(input_folder_path, output_folder_path, debug_folder_path)
bss.clear_output_folders()
bss.split()

print('All done.')
