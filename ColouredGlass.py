import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageGrab, ImageTk, Image
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

#arrays to covert between LMS and RGB color spaces
LMSTransform = np.array([[0.0841456, 0.708538, 0.148692], 
                            [-0.0767272, 0.983854, 0.0817696], 
                            [-0.0192357, 0.152575, 0.876454]])

RGBTransform = np.array([[ 7.13446359, -5.02296653, -0.74175221],
                            [ 0.55135431,  0.64315559, -0.15354206],
                            [ 0.06060046, -0.2222019,  1.15141075]])

def calcCorrect(number, strength):
    interp = np.interp(strength,[0,100],[0,number])
    return interp

def capture():
    protanTransform = np.array([[calcCorrect(1,100-stronk.get()),calcCorrect(1.05118294,stronk.get()),calcCorrect(-0.05116099,stronk.get())],
                                [0,1,0],
                                [0,0,1]])
        
    #array to manipulate color data based on the difference between normal and deficient vision
    compensatorArray = np.array([[calcCorrect(1,100-stronk.get()), 0, 0], 
                                    [calcCorrect(0.7,stronk.get()), 1, 0], 
                                    [calcCorrect(0.7,stronk.get()), 0, 1]])

    totalTransform = np.matmul(LMSTransform, np.matmul(protanTransform, RGBTransform))
    #bounding box is 4 integers, going from :left x, top y, right x, bottom y
    box = (window.winfo_rootx(), window.winfo_rooty(), window.winfo_width()+window.winfo_rootx(), window.winfo_height()+window.winfo_rooty()-captureButton.winfo_height())
    im = ImageGrab.grab(bbox=box)
    imArr = np.asarray(im)
    im.close()
    
    procs = list()
    # This is for multiprocessing. If needed, change numProcesses to some math involving cpu_count
    cpu_count = mp.cpu_count()
    numProcesses = 1
    print("Number of CPUs available: " + str(cpu_count))
    dividedList = np.array_split(imArr,numProcesses)
    processedArray = mp.Array("B", imArr.flatten(), lock=False)
    print("Storage Size: " + str(len(imArr.flatten())))
    startRow = 0
    for i in range(numProcesses):
        x = mp.Process(target=daltonizeImage, args=(dividedList[i], totalTransform, compensatorArray, startRow, processedArray, len(imArr[0])), daemon=True)
        procs.append(x)
        x.start()
        numRows = dividedList[i].shape[0]
        startRow += numRows
    
    for proc in procs:
        proc.join()

    processedArray = np.frombuffer(processedArray, dtype=np.uint8).reshape(len(imArr), len(imArr[0]), 3)
    img = ImageTk.PhotoImage(image=Image.fromarray(processedArray, mode="RGB"))
    glass.config(image = img)
    glass.image = img
    window.update()

def daltonizeImage(pixelList, totalTransform, compensatorArray, processIndex, storageArray, width):
    """Take an image or section of an image and apply a colour correction to the input such that 
    the resultant output corrects for colour-vision deficiency while also retaining the original 
    colour information of the input"""
    """Input: pixelList: a H x W x 3 numpy array
              totalTransform: a 3 x 3 numpy array
              compensatorArray: a 3 x 3 numpy array
              processIndex: an integer
              storageArray: a multiprocessing Array
              width: an integer """
    # Convert the storageArray to a writable NumPy array view
    # in the case of multiprocessing, multiple attempts to convert the storageArray won't have adverse effect
    sharedNP = np.frombuffer(storageArray, dtype=np.uint8)
    # Convert the input list of pixels into linear space (a.k.a. convert 0-255 into 0-1, with gamma correction) 
    linImg = linearizeImage(pixelList.astype(np.float32))

    # calculate the original image dimensions and reshape the multiprocessing Array
    # probably could have passed this in from the capture function, but the list of parameters was getting long
    numPixels = sharedNP.size // 3
    imageHeight = numPixels // width
    sharedNP = sharedNP.reshape((imageHeight, width, 3))

    # Daltonize the image
    sim = np.tensordot(linImg, totalTransform.T, axes=([2], [0]))
    diff = linImg - sim
    comp = np.tensordot(diff, compensatorArray.T, axes=([2], [0]))
    corrected = linImg + comp

    # Delinearize into normal colour space, Convert to int and flatten
    corrected = np.clip(delinearizeImage(corrected), 0, 255).astype(np.uint8)
    flatResult = corrected.reshape(-1, 3)

    # Calculate starting index in 1D shared array
    rowEnd = processIndex + pixelList.shape[0]
    sharedNP[processIndex:rowEnd, :, :] = corrected
    
# Clear button removes the image
def clear():
    glass.config(image = None)
    glass.image = None
    window.update()

def linearizeImage(image):
    """Vectorized sRGB to linear RGB conversion (input: float32 array, 0-255)"""
    image = image / 255.0
    linear = np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4
    )
    return linear

def delinearizeImage(image):
    """Vectorized linear RGB to sRGB conversion (output: uint8 array, 0-255)"""
    srgb = np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * (image ** (1 / 2.4)) - 0.055
    )
    srgb = np.clip(srgb * 255.0, 0, 255)
    return srgb.astype(np.uint8)

if __name__ == '__main__':
    window = tk.Tk()
    window.title("Coloured Glass")
    width = 800
    height = 600
    centerx = (window.winfo_screenwidth()//2) - (width//2)
    centery = (window.winfo_screenheight()//2) - (height//2)

    window.geometry(f'600x400+{centerx}+{centery}')
    window.attributes('-transparentcolor', 'magenta')

    buttonHolder = tk.Frame(window, height=2)
    buttonHolder.pack(side='bottom', fill='both')

    glass = tk.Label(window)
    glass.config(bg='magenta', borderwidth=0)
    glass.pack(side='top', expand=True, fill='both')

    captureButton = tk.Button(window, text='Capture', background='light grey')
    captureButton.pack(in_=buttonHolder, side='left', fill = 'both', expand=True)

    clearButton = tk.Button(window, text='Clear', background='light grey')
    clearButton.pack(in_=buttonHolder, side='right', fill = 'both', expand=True)

    pdt = tk.StringVar()
    deficiencySelector = tk.ttk.Combobox(window, textvariable = pdt)
    deficiencySelector['values'] = (' Protanopia (Red-Green; Red Weak)', 
                                    ' Deuteranopia (Red-Green; Green Weak)',
                                    ' Tritanopia (Blue-Yellow)')
    deficiencySelector.pack(in_=buttonHolder, side='left', fill = 'both', expand=True)
    deficiencySelector.current(0)
    deficiencySelector['state'] = 'readonly'

    stronk = tk.IntVar(value=100)
    strengthSlider = tk.Scale(window, from_=0, to=100, orient='horizontal', variable=stronk)
    strengthSlider.pack(in_=buttonHolder, side='right', fill = 'both', expand=True)

    captureButton.config(command=capture)
    clearButton.config(command=clear)

    window.mainloop()