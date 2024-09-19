import unittest
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



class Scan2Pdf:
    def __init__(self, input_folder_path, output_folder_path, debug_folder_path=None):
        self.ifp = input_folder_path
        self.ofp = output_folder_path
        self.dfp = debug_folder_path

        filenames = os.listdir(self.ifp)
        # filenames = ['010.jpg']
        filenames = sorted(filenames)
        filenames = [filename for filename in filenames if filename[-3:].lower() == 'jpg']

        self.n_files = len(filenames)
        self.in_filenames = []
        self.out_filenames = []
        self.hist_filenames = []
        self.Sobel_filenames = []
        self.Hough_filenames = []
        self.HoughColor_filenames = []
        self.CenterLine_filenames = []
        self.ExtractBox_filenames = []
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            self.in_filenames.append(os.path.join(self.ifp, filename))
            self.out_filenames.append(os.path.join(self.ofp, filename))
            self.hist_filenames.append(os.path.join(self.dfp, name + '_1_hist' + '.jpg'))
            self.Sobel_filenames.append(os.path.join(self.dfp, name + '_2_Sobel' + '.jpg'))
            self.Hough_filenames.append(os.path.join(self.dfp, name + '_3_Hough' + '.jpg'))
            self.HoughColor_filenames.append(os.path.join(self.dfp, name + '_4_HoughColor' + '.jpg'))
            self.CenterLine_filenames.append(os.path.join(self.dfp, name + '_5_CenterLine' + '.jpg'))
            self.ExtractBox_filenames.append(os.path.join(self.dfp, name + '_6_ExtractBox' + '.jpg'))
        
        # array for color image
        self.img = None
        # text detection results
        self.det = None

        # Color & line width
        # left & right extract boxes for pdf generation
        self.ebc, self.eblw = (0, 255, 0), 5
        # text-detected boxes
        self.dbc, self.dblw = (255, 255, 0), 2
        # histogram edges
        self.hec, self.helw = (255, 0, 0), 2
        # found center line
        self.clc, self.cllw = (0, 0, 255), 5
        # central region
        self.crc, self.crlw = (0, 255, 255), 3

    def clear_output_folder(self):
        filenames = os.listdir(self.ofp)
        for filename in filenames:
            os.remove(os.path.join(self.ofp, filename))
    
    def clear_debug_folder(self):
        filenames = os.listdir(self.dfp)
        for filename in filenames:
            os.remove(os.path.join(self.dfp, filename))
    
    def open_image(self, i):
        filename = self.in_filenames[i]
        self.img = cv2.imread(filename)
    
    def detect_text(self, i):
        if not self.is_image_loaded():
            raise(RuntimeError)

        self.det = pytesseract.image_to_data(self.img, output_type=Output.DICT)
    
    def close_image(self):
        self.img = None
    
    def is_image_loaded(self):
        return self.img is not None
    
    def get_img_width_height(self):
        if not self.is_image_loaded():
            raise(RuntimeError)

        height, width, _ = self.img.shape
        return width, height
    
    def check_single_page(self, i, save_debug_image=False):
        if not self.is_image_loaded():
            raise(RuntimeError)

        width, height = self.get_img_width_height()
        img = self.img.copy()

        is_single = False
        if width < height: # single page
            is_single = True
            if save_debug_image:
                # draw extract box
                eb = Box(0, width, 0, height)
                eb.draw(img, self.ebc, self.eblw)
                # save image as output file
                cv2.imwrite(self.ExtractBox_filenames[i], img)
            cv2.imwrite(self.out_filenames[i], self.img)
        return is_single

    def find_center_line_by_hist(self, k, save_debug_image=False, n_bin=27):
        if not self.is_image_loaded():
            raise(RuntimeError)

        width, height = self.get_img_width_height()
        img = self.img.copy()

        # 이미지에서 텍스트 추출
        d = self.det

        # 각 박스의 좌우 x 좌표 계산
        x_centers = []
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if d['level'][i] >= 2:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                Box(x, x + w, y, y + h).draw(img, self.dbc, self.dblw)
                x_centers.append(x)
                x_centers.append(x + w)
        
        # 히스토그램 생성
        bin_size = int(width / n_bin)
        n_exclude = int(n_bin * 0.4)
        hist, bin_edges = np.histogram(x_centers, bins=np.arange(0, width + bin_size, bin_size))
        for bin_edge in bin_edges[n_exclude:-(n_exclude + 1)]:
            cv2.line(img, (int(bin_edge), 0), (int(bin_edge), height), self.hec, self.helw)

        # 가장 적은 텍스트가 있는 구간을 찾음 (이곳이 중앙선일 가능성)
        min_bin_index = np.argmin(hist[n_exclude:-n_exclude]) + n_exclude # 양쪽 하나씩 제외 (페이지 좌우 마진 제외하고 가장 텍스트가 적은 구간)
        center_line = bin_edges[min_bin_index] + int(bin_size / 2)

        # histogram을 이미지에 덮어 씌우기
        plt_dpi = 100
        fig = plt.figure(figsize=(4, 3), dpi=plt_dpi, frameon=True)
        act_dpi = fig.get_dpi()
        res_ratio = int(act_dpi / plt_dpi)
        # print(f'(width, height) = {plt.gcf().canvas.get_width_height()}')
        plt.bar(bin_edges[:-1] + bin_size/2, hist, width = bin_size*0.8)
        # plt.xticks(bin_edges)
        plt.xlim([0, width])
        plt.tight_layout()
        fig.canvas.draw()
        plt_img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        plt_width, plt_height = plt.gcf().canvas.get_width_height()
        plt_img = plt_img.reshape(plt_height*res_ratio, plt_width*res_ratio, 3)
        plt.close() # 플롯을 닫아 메모리 누수 방지
        # resized_plot = cv2.resize(plt_img, (600, 400))
        resized_plot = plt_img
        x_offset, y_offset = 10, 15 # 덮어씌울 위치 설정 (예: 좌상단)
        img[y_offset:y_offset + resized_plot.shape[0], x_offset:x_offset + resized_plot.shape[1]] = resized_plot
        cv2.rectangle(img, (x_offset, y_offset), (plt_width*2 + x_offset, plt_height*2 + y_offset), (0, 0, 0), 3)

        # 중앙선 그리기
        cv2.line(img, (int(center_line), 0), (int(center_line), height), self.clc, self.cllw)

        # 디버그용 이미지로 저장
        if save_debug_image:
            cv2.imwrite(self.hist_filenames[k], img)

        # print(f'{hist=}\n{hist[n_exclude:-n_exclude]=}\n{bin_edges=}\n{center_line=}')

        return center_line

    def find_center_line_by_Hough(self, k, save_debug_image=False):
        if not self.is_image_loaded():
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
        _, binary = cv2.threshold(normalized_sobelx, 10, 255, cv2.THRESH_BINARY)
        
        # 중앙 부분을 기준으로만 수직선을 찾기 위해 이미지 중앙 영역만 남김
        # height, width = binary.shape

        central_region_left = int(width * 0.40)
        central_region_right = int(width * 0.60)
        crl_ratio = central_region_left / width
        crr_ratio = central_region_right / width
        central_region = binary[:, central_region_left:central_region_right]
        cv2.rectangle(img, (central_region_left, 0), (central_region_right, height), self.crc, self.crlw)

        # 수직선 감지 (Hough Line Transform 사용)
        lines = cv2.HoughLinesP(central_region, rho=1, theta=np.pi/180, threshold=100, minLineLength=80, maxLineGap=15)

        fold_position = None
        fold_pos_ratio = None

        if lines is not None:
            try:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        if abs(x1 - x2) < 10:  # 거의 수직인 선만 추출
                            fold_position = x1 + central_region_left  # 중앙 영역 기준 좌표 변환
                            fold_pos_ratio = fold_position / width
                            # 빨간색 수직선 그리기
                            cv2.line(img, (fold_position, 0), (fold_position, height), self.clc, self.cllw)
                            raise Exception('break')
            except Exception:
                pass

        # 결과 이미지 저장
        if save_debug_image:
            # cv2.imwrite(self.Sobel_filenames[k], normalized_sobelx)
            # cv2.imwrite(self.Hough_filenames[k], binary)
            cv2.imwrite(self.HoughColor_filenames[k], img)

        return fold_position
    
    def select_center_line(self, xc_hi, xc_Hg):
        width, height = self.get_img_width_height()
        x_half = int(width / 2)
        xc = xc_hi if abs(xc_hi - x_half) < abs(xc_Hg - x_half) else xc_Hg
        return xc
    
    def draw_center_lines(self, k, xc_hi, xc_Hg, xc):
        img = self.img.copy()
        width, height = self.get_img_width_height()
        x_half = int(width / 2)

        cv2.line(img, (x_half, 0), (x_half, height), self.dbc, 7)
        cv2.line(img, (xc_hi, 0), (xc_hi, height), self.ebc, 7)
        cv2.line(img, (xc_Hg, 0), (xc_Hg, height), self.hec, 7)
        cv2.line(img, (xc, 0), (xc, height), self.clc, 3)

        cv2.imwrite(self.CenterLine_filenames[k], img)


    def draw_text_regions(self, k, xc, margin=50, min_rate=0.6, save_debug_image=False):
        if not self.is_image_loaded():
            raise(RuntimeError)

        width, height = self.get_img_width_height()
        img = self.img.copy()

        # 이미지에서 텍스트 추출
        d = self.det
        x0, y0, w0, h0 = d['left'][0], d['top'][0], d['width'][0], d['height'][0]
        fullpage = Box(x0, x0 + w0, y0, y0 + h0)
        left_half_page = Box(fullpage.left, xc, fullpage.top, fullpage.bottom)
        right_half_page = Box(xc, fullpage.right, fullpage.top, fullpage.bottom)
        default_left_extract_page = Box(fullpage.left + margin, xc - margin, fullpage.top + margin, fullpage.bottom - margin)
        default_right_extract_page = Box(xc + margin, fullpage.right - margin, fullpage.top + margin, fullpage.bottom - margin)

        det_left = []
        det_right = []
        
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if d['level'][i] >= 2:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                detected = Box(x, x + w, y, y + h)
                detected.draw(img, self.dbc, self.dblw)
                if detected.right < xc:
                    det_left.append(detected.lrtb)
                elif detected.left > xc:
                    det_right.append(detected.lrtb)
                # print(detected)
        
        det_left = np.array(det_left) # num_detected x 4
        det_right = np.array(det_right)

        text_region_left = default_left_extract_page
        if len(det_left) > 0:
            text_region_left = Box(
                max(min(det_left[:,0]) - margin, left_half_page.left),
                min(max(det_left[:,1]) + margin, left_half_page.right),
                # max(min(det_left[:,2]) - margin, left_half_page.top),
                # min(max(det_left[:,3]) + margin, left_half_page.bottom),
                default_left_extract_page.top,
                default_left_extract_page.bottom,
                )
            if text_region_left.width < fullpage.width / 2 * min_rate or text_region_left.height < fullpage.height * min_rate:
                text_region_left = default_left_extract_page
            
        if len(det_right) == 0:
            text_region_right = default_right_extract_page
        else:
            text_region_right = Box(
                max(min(det_right[:,0]) - margin, right_half_page.left),
                min(max(det_right[:,1]) + margin, right_half_page.right),
                # max(min(det_right[:,2]) - margin, right_half_page.top),
                # min(max(det_right[:,3]) + margin, right_half_page.bottom),
                default_right_extract_page.top,
                default_right_extract_page.bottom,
                )
            if text_region_right.width < fullpage.width / 2 * min_rate or text_region_right.height < fullpage.height * min_rate:
                text_region_right = default_right_extract_page

        text_region_left.draw(img, self.ebc, self.eblw)
        text_region_right.draw(img, self.ebc, self.eblw)

        if save_debug_image:
            cv2.imwrite(self.ExtractBox_filenames[k], img)

        extracted_region_left = self.img[text_region_left.top:text_region_left.bottom, text_region_left.left:text_region_left.right]
        extracted_region_right = self.img[text_region_right.top:text_region_right.bottom, text_region_right.left:text_region_right.right]

        path, filename = os.path.split(self.out_filenames[k])
        file, ext = os.path.splitext(filename)
        filename_left = os.path.join(path, filename + '-1' + ext)
        filename_right = os.path.join(path, filename + '-2' + ext)

        cv2.imwrite(filename_left, extracted_region_left)
        cv2.imwrite(filename_right, extracted_region_right)
    
    def convert(self):
        for k in range(self.n_files):
            filename = self.in_filenames[k]
            print('----- ' + filename + ' -----')
            self.open_image(k)
            is_single = self.check_single_page(k, save_debug_image=False)
            if not is_single:
                self.detect_text(k)
                xc_hi = self.find_center_line_by_hist(k, save_debug_image=False)
                xc_Hg = self.find_center_line_by_Hough(k, save_debug_image=False)
                xc = self.select_center_line(xc_hi, xc_Hg)
                # self.draw_center_lines(k, xc_hi, xc_Hg, xc)
                self.draw_text_regions(k, xc)
                print(f'{xc_hi=}, {xc_Hg=} : {xc=}')
            self.close_image()




class TestBookScan(unittest.TestCase):
    def test_main(self):
        input_folder_path = 'input_folder/'
        output_folder_path = 'output_folder/'
        debug_folder_path = 'debug_folder/'

        s2p = Scan2Pdf(input_folder_path, output_folder_path, debug_folder_path)

        s2p.clear_output_folder()
        s2p.clear_debug_folder()
        s2p.convert()

if __name__ == '__main__':
    unittest.main()

