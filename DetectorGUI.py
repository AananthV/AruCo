'''
    Author Name : Aananth V
    Domain: Signal Processing and ML
    Sub-Domain: Image Processing
    Functions: selectImage, detectMarkers, updateImage
    Global Variables: img, panel, root, detector
'''

import numpy as np
import cv2
import tkinter as tk
import tkinter.filedialog as FileDialog
from PIL import Image, ImageTk
from markerdetector import Detector

'''
    Function Name: selectImage
    Logic:
        - Opens a filedialog using tkinter.filedialog and updates global img
        - Calls updateImage to update the panel
    Example Call:
        selectImage()
'''
def selectImage():
    global img
    path = FileDialog.askopenfilename()
    if len(path) > 0:
        img = cv2.imread(path)
    updateImage(img)

'''
    Function Name: detectMarkers
    Logic:
        - Checks if global img exists, and if it does, detects AruCo markers and updates the panel.
    Example Call:
        detectMarkers()
'''
def detectMarkers():
    global img, message

    # Check if image exists.
    if img is None:
        message.configure(text="Select Image first!")
    else:
        # Find markers.
        _img = detector.detect(img)

        # Error message if images are not of same dimension.
        updateImage(_img)

'''
    Function Name: updateImage
    Inputs: img (OpenCV image, numpy array)
    Logic:
        - Updates the panel to display img
    Example Call:
        img = cv2.imread("./Sample/img1.png")
        updateImage(img)
'''
def updateImage(img):
    global root, panel

    # Check if image exists.
    if img is None:
        return

    # OpenCV stores images in BGR format. Convert to RGB.
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use PIL to obtain Tkinter format images.
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    # Check if panel is un initialised.
    if panel is None:
        panel = tk.Label(image=image)
        panel.image = image
        panel.pack(side="left", padx=10, pady=10)

    # If panel is initialised then simply update it.
    else:
        panel.configure(image=image)
        panel.image = image

# Declare Global Variables
img = None
panel = None

# Initialise GUI
root = tk.Tk(screenName=None, baseName=None, className=' AruCo', useTk=1)

# Initialise Detector
detector = Detector()

detectButton = tk.Button(root, text="Detect", command=detectMarkers)
detectButton.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

sbtn = tk.Button(root, text="Select Image", command=selectImage)
sbtn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

message = tk.Label(root, text="Select Images and detect markers")
message.pack(side="bottom", padx=10, pady=10)

root.mainloop()
