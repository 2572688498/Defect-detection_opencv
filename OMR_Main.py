import os
import cv2
import numpy as np


class DefectDetection:
    def __init__(self, image_size=(500, 500), threshold_value=130, max_value=255, inv_threshold_value=50,
                 inv_max_value=255,
                 canny_threshold1=100, canny_threshold2=70, con_color=(0, 0, 255), con_thickness=1,
                 font=cv2.FONT_HERSHEY_SIMPLEX, output_folder='Output Images',white=(255, 255, 255),black = (0, 0, 0),green= (0, 255, 0),red= (0, 0, 255),stack_img_size= (200, 200)):
        # Initialize parameters
        self.WHITE=white
        self.BLACK=black
        self.GREEN=green
        self.RED=red
        self.STACK_IMG_SIZE=stack_img_size
        self.IMAGE_SIZE = image_size
        self.THRESHOLD = threshold_value
        self.THRESHOLD_VALUE = threshold_value
        self.MAX_VALUE = max_value
        self.INV_THRESHOLD_VALUE = inv_threshold_value
        self.INV_MAX_VALUE = inv_max_value
        self.THRESHOLD1 = canny_threshold1
        self.THRESHOLD2 = canny_threshold2
        self.CON_COLOR = con_color
        self.CON_THICKNESS = con_thickness
        self.font = font
        self.output_folder = output_folder
    #处理图片并存储
    def process_image(self, file_path):
        # 1:读数据
        imageOri = cv2.imread(path)
        # 2：灰度图
        image = cv2.cvtColor(imageOri, cv2.COLOR_BGR2GRAY)
        # 3:resize
        imageOri = cv2.resize(imageOri, self.IMAGE_SIZE)
        image = cv2.resize(image, self.IMAGE_SIZE)
        # 4:THRESHOLD操作
        ret, thresh_basic = cv2.threshold(image, self.THRESHOLD_VALUE, self.MAX_VALUE, cv2.THRESH_BINARY)
        # 5:形态学操作
        kernel = np.ones((3, 3), np.uint8)
        img_erosion = cv2.erode(thresh_basic, kernel, iterations=1)
        ret, thresh_inv = cv2.threshold(img_erosion, self.INV_THRESHOLD_VALUE, self.INV_MAX_VALUE, cv2.THRESH_BINARY_INV)
        # 6:Canny边缘检测
        edged = cv2.Canny(img_erosion, self.THRESHOLD1, self.THRESHOLD2)
        # 7:轮廓
        contours, h = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Get number of contours (excluding the background contour)
        num_of_con = str(len(contours) - 1)

        # 情况一：有缺陷
        if int(num_of_con) != 0:
            for i in range(int(num_of_con)):
                highlighted_img = cv2.drawContours(imageOri, contours, i, self.CON_COLOR, self.CON_THICKNESS)

            highlighted_img = cv2.putText(highlighted_img, f'Approximately {num_of_con} defect(s) detected', (5, 15),
                                              self.font, 0.5, self.GREEN, 1, cv2.LINE_AA)
        # 情况二：无缺陷
        else:
            highlighted_img = cv2.putText(imageOri, 'Unable to detect defects!', (5, 15), self.font, 0.5, self.RED, 2,
                                              cv2.LINE_AA)
        if file.endswith('.jpg'):
            cv2.imwrite('Output Images/{}_DEFECTS_HIGHLIGHTED.jpg'.format(file.split('.')[0]), highlighted_img)  # 存储为JPG格式的图片
        else:
            cv2.imwrite('Output Images/{}_DEFECTS_HIGHLIGHTED.jpeg'.format(file.split('.')[0]), highlighted_img)  # 存储为JPEG格式的图片

    def display_processed_images(self,para):
        # Display all processed images in the output folder
        image_files = [f for f in os.listdir(self.output_folder) if f.endswith(('.jpg','.jpeg'))]
        for file in image_files:
            image_path = 'Output Images/'+file
            img = cv2.imread(image_path)
            #仅显示jpg格式
            if para==1:
                if file.endswith('.jpg'):
                    cv2.imshow(f'JPG Image: {file}', img)
            #仅显示jpeg格式
            else:
                if file.endswith('.jpeg'):
                    cv2.imshow(f'JPEG Image: {file}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detector = DefectDetection()
    '''
    # 检测显示jpeg格式图片
    for i in range(1, 5):
        file = f'x{i}.jpeg'
        path = 'images/' + file
        detector.process_image(path)
    detector.display_processed_images(0)
    '''
    # 检测显示jpg格式图片
    for i in range(1, 11):
        file = f's{i}.jpg'
        path = 'images/' + file
        detector.process_image(path)
    detector.display_processed_images(1)


