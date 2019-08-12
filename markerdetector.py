'''
    Author Name : Aananth V
    Domain: Signal Processing and ML
    Sub-Domain: Image Processing
'''

import numpy as np
import cv2
import itertools

'''
    Class Name: Detector
    - Helper Class which is used to detect AruCo markers in a given image.
'''
class Detector:
    '''
        Function Name: inRange
        Inputs: value, min, max
        Outputs: Boolean Value
        Logic:
            - Returns true if value is in [min, max]
        Example Call:
            inRange(2, 1, 3) # True
    '''
    def inRange(self, value, min, max):
        return value >= min and value <= max

    '''
        Function Name: ptDist
        Inputs: self, pt1, pt2, n
        Output: distance
        Logic:
            - Computes the distace between two 'n' dimensional vectors 'pt1' and 'pt2'
        Example Call:
            To find the distace between the points (3,0) and (0,4) on the 2D Plane
                pt1 = [3, 0]
                pt2 = [0, 4]
                distance = self.ptDistance(pt1, pt2, 2)
            The computed distance = 5.
    '''
    def ptDist(self, pt1, pt2, n = 2):
        distance = 0
        for i in range(n):
            distance += (pt1[0][i] - pt2[0][i])**2
        return distance**0.5

    '''
        Function Name: equivalence_classes
        Inputs: self, iterable, relation
        Output: class
        Logic:
            - Splits the 'iterable' into equivalance classes on the 'relation'
            - More on Equivalence Classes: https://en.wikipedia.org/wiki/Equivalence_class
        Example Call:
            In this example the Set S = {0, 1, 2, 3, 4} is split into equivalence classes
            on the Relation R = {aRb: (a - b) MOD 2 = 0}
                S = [i for i in range(5)]
                R = lambda i, j: (i - j) % 2 == 0
                EC = self.equivalence_classes(S, R)
            The Equivalence Classes EC so formed are {0, 2, 4} and {1, 3}
    '''
    def equivalence_classes(self, iterable, relation):
        classes = []
        for o in iterable:  # for each object
            # find the class it is in
            found = False
            for c in classes:
                if relation(next(iter(c)), o):  # is it equivalent to this class?
                    c.append(o)
                    found = True
                    break
            if not found:  # it is in a new class
                classes.append([o])
        return classes

    '''
        Function Name: imageContours
        Inputs:
            1. Image
                - A single channel image.

            2. minMarkerPerimeterRate, maxMarkerPerimeterRate
                -   These parameters determine the minimum and maximum size of a
                    marker, concretely the maximum and minimum marker perimeter.
                -   They are not specified in absolute pixels values, instead they are
                    specified relative to the maximum dimension of the input image.

            3. polygonalApproxAccuracyRate
                -   A polygonal approximation is applied to each candidate and
                    only those that approximate to a square shape are accepted.
                -   This value determines the maximum error that the polygonal
                    approximation can produce.

            4. minCornerDistanceRate
                -   Minimum distance between any pair of corners in the same marker.
                -   It is expressed relative to the marker perimeter.
                -   Minimum distance in pixels is Perimeter * minCornerDistanceRate.

            5. minMarkerDistanceRate
                -   Minimum distance between any pair of corners from two different
                    markers.
                -   It is expressed relative to the minimum marker perimeter of
                    the two markers.
                -   If two candidates are too close, they are grouped together.
                -   The biggest candidate in a group is alone chosen.

        Outputs: contours
        Example Call:
            imageContours(image)
    '''
    def imageContours(self,
        thresh, minMarkerPerimeterRate = 0.05, maxMarkerPerimeterRate = 4,
        polygonalApproxAccuracyRate = 0.05, minCornerDistanceRate = 0.05,
        minMarkerDistanceRate = 0.05):

        #Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Find Approx Polygon of Each Contour
        contours = [[cnt, cv2.approxPolyDP(cnt, polygonalApproxAccuracyRate*cv2.arcLength(cnt, True), True)] for cnt in contours]

        #Filter by perimeter
        minMarkerPerimeter = max(thresh.shape)*minMarkerPerimeterRate
        maxMarkerPerimeter = max(thresh.shape)*maxMarkerPerimeterRate
        perimeterFilteredContours = filter(
            lambda cnt: self.inRange(cv2.arcLength(cnt[0], True), minMarkerPerimeter, maxMarkerPerimeter),
            contours
        )

        #Filter by number of corners (To filter quadrilaterals)
        polyFilteredContours = filter(
            lambda cnt: len(cnt[1]) == 4,
            perimeterFilteredContours
        )

        #Filter by minimum corner distance (To ignore extremely skewed quadrilaterals)
        def minPairCornerDistance(polygon, minDist):
            for pair in itertools.combinations(polygon, r=2):
                if self.ptDist(*pair) < minDist:
                    return False
            return True

        cornerDistanceFilteredContours = filter(
            lambda cnt: minPairCornerDistance(
                cnt[1],
                minCornerDistanceRate*cv2.arcLength(cnt[0], True)
            ),
            polyFilteredContours
        )

        #FIlter by minimum corner distance (to prevent overlaps)
        def minPairMarkerDistance(contourPair, minDist):
            for cornerPairs in itertools.product(contourPair[0][1], contourPair[1][1]):
                if self.ptDist(*cornerPairs) < minDist:
                    return True
            return False

        markerDistanceFilteredContours = self.equivalence_classes(
            cornerDistanceFilteredContours,
            lambda c1, c2: minPairMarkerDistance(
                [c1, c2],
                minMarkerDistanceRate*min(cv2.arcLength(c1[0], True), cv2.arcLength(c2[0], True))
            )
        )

        return np.array([sorted(i[0], key=cv2.contourArea, reverse=True)[0] for i in markerDistanceFilteredContours])

    '''
        Function Name: extractPerspective
        Inputs:
            1. image
            2. contours
                - contours of the image which can be approximated to a quadrilateral
            3. polygonalApproxAccuracyRate
                - Same as in imageContours
        Outputs:
            extracted
                - An array containing the extractedPerspective of the given contours\
        Logic:
            - Accepts an image and its quadrilateral contours.
            - Wraps the contours to 400x400 squares.
            - Uses Otsu's binarisation to convert the images to absolute binary images.
            - Returns the images.
        Example Call:
            extractPerspective(image, contours)
    '''
    def extractPerspective(self, image, contours, polygonalApproxAccuracyRate = 0.05):
            extracted = []
            for cnt in contours:
                # Find Square Edges
                approx = cv2.approxPolyDP(cnt, polygonalApproxAccuracyRate*cv2.arcLength(cnt, True), True)
                if len(approx) == 4:
                    # Get points to map to.
                    pts2 = np.float32([[0,0],[400,0],[400,400],[0,400]])

                    # Get Perspective Transform and Wrap Perspective
                    M = cv2.getPerspectiveTransform(approx.astype(np.float32), pts2)
                    dst = cv2.warpPerspective(image, M, (400,400))

                    # Blur and apply Otsu's binarisation
                    blur = cv2.GaussianBlur(dst, (3, 3), 0)
                    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    # Append to extracted.
                    extracted.append(otsu)

            # Return Extracted
            return extracted

    '''
        Function Name: binArrayToDecimal
        Inputs: array
            -   An Array whos elements are 1s and 0s representing a binary number
                in big endian notation
        Outputs: value
            -   Decimal form of the binary number
        Example Call:
            bin = [1, 0, 1]
            binArrayToDecimal(bin) # Outputs 5;
    '''
    def binArrayToDecimal(self, array):
        value = 0
        for i in range(len(array)):
            value += array[len(array) - i - 1]*2**i
        return value

    '''
        Function Name: decodeAruCo5x5
        Inputs: extractedMatrix
                -   A 5x5 boolean matrix representing the AruCo marker.
        Outputs: value
        Logic:
                -   Checks the hamming code to verify if the matrix is a valid
                    AruCo marker.
                -   If it is valid,
                    Find and return the marker ID
                -   Else,
                    Returns -1
        Example Call:
                decodeAruCo5x5(np.ones((5, 5))) # Returns -1 as it is invalid.
    '''
    def decodeAruCo5x5(self, extractedMatrix):
        binaryValue = np.zeros(10, dtype=np.int8)
        for row in range(5):
            data1 = extractedMatrix[row][1]
            data2 = extractedMatrix[row][3]
            parity1 = extractedMatrix[row][4]
            parity2 = extractedMatrix[row][0]
            parity3 = extractedMatrix[row][2]
            if parity1 is not data1 ^ data2:
                return -1
            if parity2 is data1:
                return -1
            if parity3 is not data2:
                return -1
            binaryValue[2*row] = data1
            binaryValue[2*row + 1] = data2
        return self.binArrayToDecimal(binaryValue)

    '''
        Function Name: feedDecoder
        Inputs: extractedMatrix
                -   A 5x5 boolean matrix representing the AruCo marker.
        Outputs: value
        Logic:
            -   Obtains value (marker ID) by feeding the extracted matrix and its
                three rotated forms to decodeAruCo5x5.
            -   If marker is invalid, returns -1.
        Example Call:
            feedDecoder(np.ones((5, 5))) # Returns -1 as it is invalid.
    '''
    def feedDecoder(self, extractedMatrix):
        matrix = extractedMatrix
        counter = 0
        value = -1
        while counter < 4:
            value = self.decodeAruCo5x5(matrix)
            if value >= 0:
                return value
            counter += 1
            matrix = np.rot90(matrix)
        return value

    '''
        Function Name: extractAruCo
        Inputs:
            1. extractedImage
                -   An image consisting solely of the AruCo marker and its black border.
            2. divisions
                -   Total number of divisions in the image, including the border bits.
            3. markerBorderBits
                -   Thickness of the AruCo border relative to each bit.
            4. perspectiveRemoveIgnoredMarginPerCell
                -   A bit of padding is removed in each cell to reduce errors.
                -   Only the centers of the cells are considered.
        Outputs: marker id (if the marker is valid).
        Logic:
            -   Loops through the cells making up the marker.
            -   Removes a bit of padding to reduce errors.
            -   Calculates the dominant color in the region and sets that
                as the cell color (White = 1, Black = 0)
            -   Obtains AruCo ID by feeding AruCo matrix into feedDecoder
        Example Call:
            extractAruCo(image)
    '''
    def extractAruCo(
        self, extractedImage, divisions = 7, markerBorderBits = 1,
        perspectiveRemoveIgnoredMarginPerCell = 0.2,
        ):

        AruCoSize = divisions - 2*markerBorderBits
        AruCo = np.zeros((AruCoSize, AruCoSize), dtype=np.bool);
        cellSize = extractedImage.shape[0]//divisions

        # Loop through divisions
        for row in range(AruCoSize):
            for col in range(AruCoSize):
                cell = extractedImage[
                    int(cellSize*(row+markerBorderBits) + cellSize*perspectiveRemoveIgnoredMarginPerCell):
                    int(cellSize*(row+markerBorderBits+1) - cellSize*perspectiveRemoveIgnoredMarginPerCell),
                    int(cellSize*(col+markerBorderBits) + cellSize*perspectiveRemoveIgnoredMarginPerCell):
                    int(cellSize*(col+markerBorderBits+1) - cellSize*perspectiveRemoveIgnoredMarginPerCell),
                ]
                AruCo[row, col] = (cell > 0).flatten().sum() > cell.shape[0]

        return self.feedDecoder(AruCo)

    '''
        Function Name: drawMarkerIds
        Inputs:
            1. image
            2. contours
            3. AruCoIds
                -   AruCoIds corresponding to the contours.
        Logic:
            -   Draws the bounding contours and the marker IDs of the AruCo markers
                if the markers are valid. (ID != -1).
        Example Call:
            drawMarkerIds(image, contours, AruCoIds)
    '''
    def drawMarkerIds(self, image, contours, AruCoIds):
        for id in range(len(AruCoIds)):
            if AruCoIds[id] == -1:
                continue
            cv2.drawContours(image, contours, id, (255,0,0), 2)
            center, radius = cv2.minEnclosingCircle(contours[id])
            center = (center[0], center[1] + radius/2)
            center = tuple([int(i) for i in center])
            cv2.putText(image, str(AruCoIds[id]), center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    '''
        Function Name: detect
        Inputs:
                1. img
                    -   The image to detect the AruCo markers in.
        Outputs:
                1. image
                    -   The original image but with detected marker boundaries and IDs
        Logic:
            -   Accepts an image and detects if it contains any 5x5 AruCo markers.
            -   Draws marker boundaries and IDs of detected markers.
        Example Call:
            image = cv2.imread("Sample/img1.png")
            detect(image)
    '''
    def detect(self, img):
        # Create a copy of original image to prevent overwriting
        image = img.copy()

        # Convert to Grayscale and apply adaptive thresholding
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 7)

        # Find and filter contours using imageContours
        contours = self.imageContours(thresh)

        # Extract the possible markers' as 400x400 binary images.
        extractedImages = self.extractPerspective(imgray, contours)

        # Obtain the AruCo ID of each extracted image (Or -1 if it isn't a AruCo Marker)
        AruCoIds = []
        for img in extractedImages:
            AruCoIds.append(self.extractAruCo(img))

        # Draw the marker IDs
        self.drawMarkerIds(image, contours, AruCoIds)

        # Return image
        return image
