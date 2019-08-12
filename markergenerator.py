'''
    Author Name : Aananth V
    Domain: Signal Processing and ML
    Sub-Domain: Image Processing
'''

import numpy as np
import cv2

'''
    Class Name: Generator
    - Helper Class which is used to generate AruCo Markers
'''
class Generator:
    '''
        Function Name: generateMarkerImage
        Input: marker (7x7 Matrix representing the AruCo Marker.)
        Output: image (of the AruCo Marker as a 500x500 numpy array.)
        Logic: Uses cv2.rectangle to draw the marker represented by the input matrix
        Example Call:
            self.generateMarkerImage(np.zeros((7, 7)));
    '''
    def generateMarkerImage(self, marker):
        image = np.ones((500, 500, 1), dtype=np.uint8)*255
        cellSize = 400//7
        for row in range(7):
            for col in range(7):
                cv2.rectangle(image, (50 + cellSize*col, 50 + cellSize*row), (50 + cellSize*(col + 1), 50 + cellSize*(row+1)), 255*marker[row, col], -1)

        return image

    '''
        Function Name: generateMarker
        Inputs: id (Marker ID: An integer between 0 and 1023).
        Output: image (of the AruCo Marker as a 500x500 numpy array.)
        Logic:
            -   Generates the marker as a 7x7 Matrix using AruCo and Hamming Code rules.
            -   Uses generateMarkerImage to then generate the image.
        Example Call:
            self.generateMarkerImage(650)
    '''
    def generateMarker(self, id):
        marker = np.zeros((7, 7))
        bin_id = list(bin(id)[2:].zfill(10))
        bin_id = [int(i) for i in bin_id]
        for row in range(5):
            marker[1+row][1+1] = bin_id[2*row]
            marker[1+row][1+3] = bin_id[2*row + 1]
            marker[1+row][1+0] = not bin_id[2*row]
            marker[1+row][1+2] = bin_id[2*row + 1]
            marker[1+row][1+4] = bin_id[2*row] ^ bin_id[2*row + 1]
        return self.generateMarkerImage(marker)
