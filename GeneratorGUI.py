'''
    Author Name : Aananth V
    Domain: Signal Processing and ML
    Sub-Domain: Image Processing
    Functions: generateMarker, updateImage
    Global Variables: panel, root, generator, id
'''

import numpy as np
import cv2
import tkinter as tk
import tkinter.filedialog as FileDialog
from PIL import Image, ImageTk
from markergenerator import Generator

'''
    Function Name: saveMarker
    Logic:
        -   Saves the marker as a png as a location obatined using tkinter.filedialog
    Example Call:
        saveMarker()
'''
def saveMarker():
    global img
    if img is not None:
        path = FileDialog.asksaveasfilename(defaultextension=".png")
        if len(path) > 0:
            cv2.imwrite(path, img)
    else:
        message.configure(text="Generate Marker First!")

'''
    Function Name: generateMarker
    Logic:
        - Extracts id from text field by id.get()
        - Checks if id is in the allowed range
        - Generates marker using Generator and updates panel using updateImage
    Example Call:
        generateMarker()
'''
def generateMarker():
    global img, id, message
    markerId = id.get()
    try:
        markerId = int(markerId)
        if not (markerId >= 0 and markerId < 1024):
            message.configure(text="Enter integral id between 0 and 1023.")
        else:
            message.configure(text="Generated!")
            img = generator.generateMarker(markerId)
            updateImage(img)
    except:
        message.configure(text="Enter integral id between 0 and 1023.")

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

# Inditialise GUI
root = tk.Tk(screenName=None, baseName=None, className=' AruCo', useTk=1)

# Declare Global Variables
img = None
panel = None
id = tk.StringVar()

# Initialise Generator
generator = Generator()

saveButton = tk.Button(root, text="Save AruCo Marker", command=saveMarker)
saveButton.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

generateButton = tk.Button(root, text="Generate AruCo Marker", command=generateMarker)
generateButton.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

idEntryFrame = tk.Frame(root)
idText = tk.Label(idEntryFrame, text="ID: ")
idEntry = tk.Entry(idEntryFrame, textvariable=id)
idText.pack(side="left", fill="both", expand="yes", padx=10, pady=10)
idEntry.pack(side="left", fill="both", expand="yes", padx=10, pady=10)
idEntryFrame.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

message = tk.Label(root, text="Select Images and find differences!")
message.pack(side="bottom", padx=10, pady=10)

root.mainloop()
