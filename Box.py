import unittest
import cv2

class Box:
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
    
    def __repr__(self):
        return f'(L, R, T, B) = ({self.left}, {self.right}, {self.top}, {self.bottom})'
    
    @property
    def width(self):
        return self.right - self.left
    
    @property
    def height(self):
        return self.bottom - self.top

    @property
    def top_left(self):
        return (self.left, self.top)
    
    @property
    def bottom_right(self):
        return (self.right, self.bottom)
    
    @property
    def lrtb(self):
        return (self.left, self.right, self.top, self.bottom)

    def draw(self, img, color=(0, 0, 0), linewidth=2):
        cv2.rectangle(img, self.top_left, self.bottom_right, color, linewidth)


class TestBox(unittest.TestCase):
    def test_box(self):
        b1 = Box(30, 450, 10, 250)
        print(b1)
        print(f'top_left = {b1.top_left}')
        print(f'bottom_right = {b1.bottom_right}')
        
        img = cv2.imread('input_folder/010.jpg')
        img = b1.draw(img, (0, 255, 0), 5)
        cv2.imshow('box test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    unittest.main()
