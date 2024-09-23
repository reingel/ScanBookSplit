import os
import sys
from dataclasses import dataclass
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
from Box import Box



class BookScanSplit:
    def __init__(self, input_folder, output_folder, debug_folder=None):
        # folders for loading input images and saving output & debug images
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.debug_folder = debug_folder

        # array for color image
        self.img = None
        # text detection results
        self.text_data = None

        # Load and filter a list of files (jpg only)
        filenames = sorted([f for f in os.listdir(self.input_folder) if f.lower().endswith('.jpg')])
        self.n_files = len(filenames)

        # Store paths for input, output, and debug images
        self.input_files = [os.path.join(self.input_folder, f) for f in filenames]
        self.output_files = [os.path.join(self.output_folder, f) for f in filenames]
        if self.debug_folder:
            self.debug_files = {
                'hist': [self._debug_file(f, '_1_hist') for f in filenames],
                'hough': [self._debug_file(f, '_2_hough') for f in filenames],
                'center': [self._debug_file(f, '_3_center') for f in filenames],
                'page': [self._debug_file(f, '_4_page') for f in filenames]
            }

        # settings for drawing style
        self.styles = { # (color, thickness)
            'page_box_style': ((0, 255, 0), 5),
            'text_box_style': ((255, 255, 0), 2),
            'histogram_edge_style': ((255, 0, 0), 2),
            'center_line_style': ((0, 0, 255), 5),
            'central_region_style': ((0, 255, 255), 3),
        }

        # parameters for finding center lines
        self.params = {
            'central_region_ratio': 0.4,
            'vertical_line_criteria': 10,
            'binarization_threshold': 10,
            'Hough_min_line_length': 80,
            'Hough_max_line_gap': 15,
            'page_margin': 50,
            'min_page_size_ratio': 0.6,
        }

    def _debug_file(self, filename, suffix):
        """Helper function to generate filenames for debug images."""
        name, ext = os.path.splitext(filename)
        return os.path.join(self.debug_folder, f"{name}{suffix}{ext}")

    def clear_output_folders(self):
        self.clear_folder(self.output_folder)
        if self.debug_folder:
            self.clear_folder(self.debug_folder)

    def clear_folder(self, folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

    def load_image(self, i):
        filename = self.input_files[i]
        self.img = cv2.imread(filename)
    
    def close_image(self):
        self.img = None
    
    def detect_text(self, i):
        if self.img is None:
            raise(RuntimeError)
        self.text_data = pytesseract.image_to_data(self.img, output_type=Output.DICT)
    
    def get_img_width_height(self):
        if self.img is None:
            raise(RuntimeError)

        height, width, _ = self.img.shape
        return width, height
    
    def check_single_page(self, i):
        if self.img is None:
            raise(RuntimeError)

        width, height = self.get_img_width_height()
        img = self.img.copy()

        is_single = False
        if width < height: # single page
            is_single = True
            if self.debug_folder:
                # draw extract box
                eb = Box(0, width, 0, height)
                eb.draw(img, *self.styles['page_box_style'])
                # save image as output file
                cv2.imwrite(self.debug_files['page'][i], img)
            cv2.imwrite(self.output_files[i], self.img)
            print('single page')
        return is_single

    def find_center_line_by_hist(self, k, n_bin=27):
        if self.img is None:
            raise(RuntimeError)

        width, height = self.get_img_width_height()
        img = self.img.copy()

        # 이미지에서 텍스트 추출
        d = self.text_data
        levels = d['level']

        # 각 박스의 좌우 x 좌표 계산
        x_centers = []
        n_boxes = len(levels)
        for i in range(n_boxes):
            if levels[i] >= 2:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                Box(x, x + w, y, y + h).draw(img, *self.styles['text_box_style'])
                x_centers.append(x)
                x_centers.append(x + w)
        
        # 히스토그램 생성
        bin_size = int(width / n_bin)
        n_exclude = int(n_bin * self.params['central_region_ratio'])
        hist, bin_edges = np.histogram(x_centers, bins=np.arange(0, width + bin_size, bin_size))
        for bin_edge in bin_edges[n_exclude:-(n_exclude + 1)]:
            cv2.line(img, (int(bin_edge), 0), (int(bin_edge), height), *self.styles['histogram_edge_style'])

        # 가장 적은 텍스트가 있는 구간을 찾음 (이곳이 중앙선일 가능성)
        min_bin_index = np.argmin(hist[n_exclude:-n_exclude]) + n_exclude # 양쪽 하나씩 제외 (페이지 좌우 마진 제외하고 가장 텍스트가 적은 구간)
        center_line = bin_edges[min_bin_index] + int(bin_size / 2)

        # # histogram을 이미지에 덮어 씌우기
        plt_img = self._create_histogram_plot(hist, bin_edges, bin_size, width)
        x_offset, y_offset = 10, 15
        img[y_offset:y_offset + plt_img.shape[0], x_offset:x_offset + plt_img.shape[1]] = plt_img
        cv2.rectangle(img, (x_offset, y_offset), (plt_img.shape[1] + x_offset, plt_img.shape[0] + y_offset), (0, 0, 0), 3)

        # 중앙선 그리기
        cv2.line(img, (int(center_line), 0), (int(center_line), height), *self.styles['center_line_style'])

        # 디버그용 이미지로 저장
        if self.debug_folder:
            cv2.imwrite(self.debug_files['hist'][k], img)

        # print(f'{hist=}\n{hist[n_exclude:-n_exclude]=}\n{bin_edges=}\n{center_line=}')

        return center_line
    
    def _create_histogram_plot(self, hist, bin_edges, bin_size, width):
        plt_dpi = 100
        fig = plt.figure(figsize=(4, 3), dpi=plt_dpi)
        act_dpi = fig.get_dpi()
        res_ratio = int(act_dpi / plt_dpi)
        plt.bar(bin_edges[:-1] + bin_size / 2, hist, width=bin_size * 0.8)
        plt.xlim([0, width])
        plt.tight_layout()

        # Convert matplotlib figure to image
        fig.canvas.draw()
        plt_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_width, plt_height = plt.gcf().canvas.get_width_height()
        plt_img = plt_img.reshape(plt_height*res_ratio, plt_width*res_ratio, 3)
        plt.close()  # Close the plot to free memory
        return plt_img

    def find_center_line_by_Hough(self, k):
        if self.img is None:
            raise(RuntimeError)

        # 이미지 그레이스케일로 변환
        img = self.img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        width, height = self.get_img_width_height()
        
        # 가우시안 블러 적용 (노이즈 제거)
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # 수직 엣지 검출을 위한 Sobel 필터 적용 (수직 방향 에지 검출)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        
        # 절대값 취하고 정규화
        abs_sobelx = np.absolute(sobelx)
        normalized_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # 이진화 (임계값 설정)
        _, binary = cv2.threshold(normalized_sobelx, self.params['binarization_threshold'], 255, cv2.THRESH_BINARY)
        
        # 중앙 부분을 기준으로만 수직선을 찾기 위해 이미지 중앙 영역만 남김
        central_region_left = int(width * self.params['central_region_ratio'])
        central_region_right = width - central_region_left
        central_region = binary[:, central_region_left:central_region_right]
        cv2.rectangle(img, (central_region_left, 0), (central_region_right, height), *self.styles['central_region_style'])

        # 수직선 감지 (Hough Line Transform 사용)
        lines = cv2.HoughLinesP(central_region, rho=1, theta=np.pi/180, threshold=100, \
            minLineLength=self.params['Hough_min_line_length'], maxLineGap=self.params['Hough_max_line_gap'])

        fold_position = None
        fold_pos_ratio = None

        if lines is not None:
            try:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        if abs(x1 - x2) < self.params['vertical_line_criteria']:  # 거의 수직인 선만 추출
                            fold_position = x1 + central_region_left  # 중앙 영역 기준 좌표 변환
                            fold_pos_ratio = fold_position / width
                            # 빨간색 수직선 그리기
                            cv2.line(img, (fold_position, 0), (fold_position, height), *self.styles['center_line_style'])
                            raise Exception('break')
            except Exception:
                pass

        # 결과 이미지 저장
        if self.debug_folder:
            cv2.imwrite(self.debug_files['hough'][k], img)

        return fold_position
    
    def select_center_line(self, k, xc_hi, xc_Hg):
        width, height = self.get_img_width_height()
        x_half = int(width / 2)
        xc = xc_hi if abs(xc_hi - x_half) < abs(xc_Hg - x_half) else xc_Hg

        img = self.img.copy()
        width, height = self.get_img_width_height()
        x_half = int(width / 2)

        cv2.line(img, (x_half, 0), (x_half, height), (255, 255, 0), 7)
        cv2.line(img, (xc_hi, 0), (xc_hi, height), (0, 255, 0), 7)
        cv2.line(img, (xc_Hg, 0), (xc_Hg, height), (255, 0, 0), 7)
        cv2.line(img, (xc, 0), (xc, height), (0, 0, 255), 3)

        if self.debug_folder:
            cv2.imwrite(self.debug_files['center'][k], img)

        return xc

    def extract_pages(self, k, xc):
        if self.img is None:
            raise(RuntimeError)

        img = self.img.copy()
        width, height = self.get_img_width_height()
        margin = self.params['page_margin']

        # 이미지에서 텍스트 추출
        d = self.text_data
        x0, y0, w0, h0 = d['left'][0], d['top'][0], d['width'][0], d['height'][0]
        fullpage = Box(x0, x0 + w0, y0, y0 + h0)
        halfpage_left = Box(fullpage.left, xc, fullpage.top, fullpage.bottom)
        halfpage_right = Box(xc, fullpage.right, fullpage.top, fullpage.bottom)
        default_page_left = Box(fullpage.left + margin, xc - margin, fullpage.top + margin, fullpage.bottom - margin)
        default_page_right = Box(xc + margin, fullpage.right - margin, fullpage.top + margin, fullpage.bottom - margin)
        min_ratio = self.params['min_page_size_ratio']
        min_width = fullpage.width / 2 * min_ratio
        min_height = fullpage.height * min_ratio

        # 중앙선 기준으로 추출된 텍스트를 좌우로 분류
        detected_textboxes_left = []
        detected_textboxes_right = []
        levels = d['level']
        n_boxes = len(levels)
        for i in range(n_boxes):
            if levels[i] >= 2:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                detected = Box(x, x + w, y, y + h)
                if detected.right < xc:
                    detected_textboxes_left.append(detected.lrtb)
                elif detected.left > xc:
                    detected_textboxes_right.append(detected.lrtb)
                detected.draw(img, *self.styles['text_box_style'])
        detected_textboxes_left = np.array(detected_textboxes_left) # size = num_detected x 4
        detected_textboxes_right = np.array(detected_textboxes_right)

        # Extract left and right pages
        page_box_left = self._get_page_box(detected_textboxes_left, default_page_left, halfpage_left, min_width, min_height, margin, 'left')
        page_box_right = self._get_page_box(detected_textboxes_right, default_page_right, halfpage_right, min_width, min_height, margin, 'right')

        # 디버그 이미지 저장
        page_box_left.draw(img, *self.styles['page_box_style'])
        page_box_right.draw(img, *self.styles['page_box_style'])
        if self.debug_folder:
            cv2.imwrite(self.debug_files['page'][k], img)

        # 추출된 페이지 저장
        path, filename = os.path.split(self.output_files[k])
        file, ext = os.path.splitext(filename)
        filename_left = os.path.join(path, file + '-1' + ext)
        filename_right = os.path.join(path, file + '-2' + ext)
        extracted_page_left = self.img[page_box_left.top:page_box_left.bottom, page_box_left.left:page_box_left.right]
        extracted_page_right = self.img[page_box_right.top:page_box_right.bottom, page_box_right.left:page_box_right.right]
        cv2.imwrite(filename_left, extracted_page_left)
        cv2.imwrite(filename_right, extracted_page_right)
   
    def _get_page_box(self, detected_textboxes, default_page, halfpage, min_width, min_height, margin, side):
        if len(detected_textboxes) > 0:
            page_box = Box(
                max(min(detected_textboxes[:, 0]) - margin, halfpage.left),
                min(max(detected_textboxes[:, 1]) + margin, halfpage.right),
                default_page.top,
                default_page.bottom,
            )
            if page_box.width < min_width or page_box.height < min_height:
                page_box = default_page
        else:
            page_box = default_page

        return page_box
 
    def split(self):
        for k in range(self.n_files):
            filename = self.input_files[k]
            print('----- ' + filename + ' -----')
            self.load_image(k)
            is_single = self.check_single_page(k)
            if not is_single:
                self.detect_text(k)
                xc_hi = self.find_center_line_by_hist(k)
                xc_Hg = self.find_center_line_by_Hough(k)
                xc = self.select_center_line(k, xc_hi, xc_Hg)
                self.extract_pages(k, xc)
                print(f'{xc_hi=}, {xc_Hg=} : {xc=}')
            self.close_image()


import unittest

class TestBookScan(unittest.TestCase):
    def test_main(self):
        input_folder= 'input_folder/'
        output_folder= 'output_folder/'
        debug_folder= 'debug_folder/'

        bss = BookScanSplit(input_folder, output_folder, debug_folder)
        # bss = BookScanSplit(input_folder, output_folder) # do not save debug images
        bss.clear_output_folders()
        # bss.clear_folder(debug_folder) # forcefully clear debug folder
        bss.split()

if __name__ == '__main__':
    unittest.main()

