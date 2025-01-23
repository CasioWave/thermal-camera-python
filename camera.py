'''
Created by: Diptanuj Sarkar (Jan 2025)

General purpose software meant for extracting useful calibrated data with error estimation from IR Thermal Cameras
Originally meant to be used with the Topdon TC001 - a relatively cheap thermal imager.
This script roughly follows this process:
1. Fetch data from camera on USB using openCV in RAW format with no RGB conversion. (Note: What RAW means depends on the drivers, please calibrate accordingly)
2. Reshapes data into XxYx2 matrix format, with each pixel having 2 values.
3. Calculates temperature matrix based on a linear (or quadratic) fit of the pixel values vs. calibrated temperature from a file

It then uses that temperature matrix for things like displaying heatmaps, taking experimental measurements, plotting, etc.

NOTE: This script is meant to collect data raw data from the camera - and thus carries out no modifications to the data.
Smoothing, contrast adjustment, etc. are all done post-facto.
The script only records the raw data, and the smoothed data is only used for display purposes.
Averaging, standard deviation, etc. all need to be done post-facto.
'''

print("Starting program...")
print("Please note that this program is meant to be used with a thermal camera that returns raw data.")
print("Please calibrate the camera accordingly.")
print("Please ensure that the camera is connected and powered on.")
print("Please ensure that the camera is the only USB camera connected. (Or set value of dev in code)")
print("Please ensure that the camera is set to the correct resolution.")
print("Calibration data should be in the form of a numpy array with dimensions (Y,X,6) for a quadratic fit and (Y,X,3) for a linear fit.")
print("The first 2 dimensions are the resolution of the camera, and the 3rd dimension is the coefficients of the fit.")
print("The coefficients are in the order: a, b, c, d, e, f for a quadratic fit and a, b, c for a linear fit.")
print("Please ensure that the calibration data is in the same folder as the script. (Or specify path in script)")
print("Please ensure that the calibration data is in the form of a numpy array.")
print("Key bindings:")
print("-> Press 'Esc' to exit the program")
print("-> Press 's' to take a snapshot")
print("-> Press 'c' to clear all tracked pixels")
print("-> Press 'h' to toggle HUD")
print("-> Press 'm' to toggle temperature statistics display")
print("-> Press 'r' to reset tracked pixels and experimental data")
print("-> Press 'q' to quit the program")
print("-> Press 'p' to print tracked pixels")
print("-> Press 't' to print experimental data")
print("-> Press 'a' to toggle recording all temperature data")
print("-> Press 'x' to toggle smoothed temperature data")
print("-> Press '1' to increase contrast")
print("-> Press '2' to decrease contrast")
print("-> Press '3' to increase blur radius")
print("-> Press '4' to decrease blur radius")
print("-> Press '5' to increase threshold for max and min temp display")
print("-> Press '6' to decrease threshold for max and min temp display")
print("-> Press '7' to cycle through interpolation methods")
print("-> Press '8' to cycle through colormaps")
print("-> Press '9' to increase smoothing order")
print("-> Press '0' to decrease smoothing order")
print("-> Press 'e' to toggle experimental data recording")

#Importing Libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import csv
from copy import deepcopy

#Important parameters
fitOrder = 1 # 1 = Linear, 2 = Quadratic

#Resolution of camera -> (y,x), y = Top to bottom, x = Left to right. In pixels
resX = 256
resY = 192

#No of frames returned by camera (TC001 returns 2 frames in one go, a contrasted image and a temperature matrix)
fNum = 2

#Interpolation algorithm for image display
interp = "cubic" #Options: "nearest", "linear", "cubic"

#Toggle between recording all temperature data or just the selected pixels
recordAll = False

#Generate plots for the temperature data?
plot = True

#Calibration fit filename
calibFile = "./calib.npy"

#Kelvin toggle (is the calibration data based on the Kelvin scale? True if yes, False if no)
kelvin = False

#Display interval
dispInt = 2 #Display every 5 frames

#Loading the file here to prevent any issues later
try:
    cdata = np.load(calibFile)
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
    print("Please ensure that the calibration file is in the same folder as the script")
    exit()

#USB Camera number
dev = 0 #Generally, it is 0 if the thermal camera is the only USB Cam connected

#Smoothing order for temperature matrix
smoothOrder = 1 # 1 for 3x3, 2 for 5x5, 3 for 7x7, etc.

#Switch between displaying smoothed and raw temperature data
smoothed = False

#Switch on/off maximum, minimum and average temperature display
tempStats = True

#Setting things up

#Setting up camera settings
cam = cv.VideoCapture(dev, cv.CAP_ANY) #CAP_ANY for widest compatibility
cam.set(cv.CAP_PROP_CONVERT_RGB, 0.0) #Forces RGB conversion off

#Setting scale of displayed image
scale = 3
newX = scale*resX
newY = scale*resY

#Controls contrast
alpha = 1.0

#Colormap to be applied, and font of data drawn on the screen
colormap = 0
font = cv.FONT_HERSHEY_SIMPLEX

#Window setup
cv.namedWindow("Thermal Camera", cv.WINDOW_GUI_NORMAL)
cv.resizeWindow("Thermal Camera", newX, newY)

#Some more default settings
bRad = 0 #Blur radius
thres = 2 #Threshold for max and min temp display

hud = True #Heads-up display

#Some global variables
trackedPixels = [] #List of pixels to track
experimentalData = [] #List of experimental data
noExp = 0 #Number of experimental datasets
expToggle = False #Toggle for experimental data recording
data = [] #List of data - holds data for one experimental run
expStartTimes = [] #List of start times for each experimental run
expEndTimes = [] #List of end times for each experimental run

#Defining some useful functions
def snap(heatmap,tMatrix):
    '''
    heatmap: numpy array containing image (RGB) information
    tMatrix: numpy array containing temperature information
    Saves the heatmap and the temperature matrix as a CSV file and an image file
    '''
    now = time.strftime("%Y%m%d-%H%M%S")
    cv.imwrite("snap-"+now+".png", heatmap)
    np.savetxt("tempSnap"+now+".csv", tMatrix, delimiter=",")
    return True

def getTemp(thdata,coords,calib=cdata,fitOrder=fitOrder,accuracy=2):
    '''
    thdata: numpy array containing temperature information
    coords: tuple containing coordinates of pixel
    calib: filename of calibration data
    fitOrder: 1 for linear, 2 for quadratic
    accuracy: number of decimal places to round off to
    Returns temperature of pixel at coords
    '''
    try:
        hi = thdata[coords[:, 0], coords[:, 1], 0]
        lo = thdata[coords[:, 0], coords[:, 1], 1]
        if fitOrder == 2:
            a = cdata[coords[:, 0], coords[:, 1], 0]
            b = cdata[coords[:, 0], coords[:, 1], 1]
            c = cdata[coords[:, 0], coords[:, 1], 2]
            d = cdata[coords[:, 0], coords[:, 1], 3]
            e = cdata[coords[:, 0], coords[:, 1], 4]
            f = cdata[coords[:, 0], coords[:, 1], 5]
            temp = a*(hi**2) + b*(lo**2) + c*hi + d*lo + e*hi*lo + f
        else:
            a = cdata[coords[:, 0], coords[:, 1], 0]
            b = cdata[coords[:, 0], coords[:, 1], 1]
            c = cdata[coords[:, 0], coords[:, 1], 2]
            temp = a*hi + b*lo + c
        
        if kelvin:
            pass
        else:
            temp += 273.15
    except IndexError as e:
        print(f"IndexError: {e}")
        return np.zeros(coords.shape[0])
    return np.round(temp, accuracy)

#Vectorised version of tempSmooth
#Assuming thdata is a NumPy array and getTemp is vectorizable
def tempSmooth(thdata, coor, order=smoothOrder, calib=cdata, fitOrder=fitOrder, accuracy=2):
    '''
    thdata: numpy array containing temperature information
    coor: tuple containing coordinates of pixel
    order: 1 for 3x3, 2 for 5x5, 3 for 7x7, etc.
    calib: calibration data
    fitOrder: 1 for linear, 2 for quadratic
    accuracy: number of decimal places to round off to
    Returns average temperature of pixels around coor within the specified order
    '''
    y, x = coor
    y_min, y_max = max(0, y - order), min(thdata.shape[0], y + order + 1)
    x_min, x_max = max(0, x - order), min(thdata.shape[1], x + order + 1)
    
    sub_thdata = thdata[y_min:y_max, x_min:x_max]
    indices = np.indices(sub_thdata.shape[:2]).reshape(2, -1).T + [y_min, x_min]
    
    temps = getTemp(thdata, indices, calib, fitOrder, accuracy)
    avg_temp = np.mean(temps)
    
    return round(avg_temp, accuracy)

def drawData(coor,temp,imdata,scale=scale):
    '''
    coor: tuple containing coordinates of pixel
    temp: temperature of pixel
    imdata: numpy array containing image (RGB) information
    scale: scale of displayed image
    Draws a crosshair and temperature data on the image
    '''
    y = coor[0]
    x = coor[1]
    # draw crosshairs
    cv.line(imdata,(x,y+20),\
    (x,y-20),(255,255,255),2) #vline
    cv.line(imdata,(x+20,y),\
    (x-20,y),(255,255,255),2) #hline

    cv.line(imdata,(x,y+20),\
    (x,y-20),(0,0,0),1) #vline
    cv.line(imdata,(x+20,y),\
    (x-20,y),(0,0,0),1) #hline
    #show temp
    cv.putText(imdata,str(temp)+' K', (x+10, y-10),\
    font, 0.45,(0, 0, 0), 2, cv.LINE_AA)
    cv.putText(imdata,str(temp)+' K', (x+10, y-10),\
    font, 0.45,(0, 255, 255), 1, cv.LINE_AA)
    return True

# Assuming thdata is a NumPy array and getTemp is vectorizable
def calculate_temperature_matrix(thdata, resY, resX, smoothed, smoothOrder):
    '''
    thdata: numpy array containing temperature information
    resY: resolution in the y-direction
    resX: resolution in the x-direction
    smoothed: boolean to toggle smoothed data
    smoothOrder: order of smoothing
    Returns the temperature matrix
    Neatly wraps the getTemp and tempSmooth functions, vectorized
    '''
    indices = np.indices((resY, resX)).reshape(2, -1).T
    if smoothed:
        tMatrixSmooth = np.array([tempSmooth(thdata, (i, j)) for i, j in indices]).reshape(resY, resX)
        return tMatrixSmooth
    else:
        return getTemp(thdata, indices, cdata, fitOrder).reshape(resY, resX)

def clickEvent(event, x, y, flags, params):
    '''
    event: event type
    x: x-coordinate of mouse click
    y: y-coordinate of mouse click
    flags: flags
    params: parameters
    Records coordinates of mouse click
    '''
    global trackedPixels
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')
        trackedPixels.append((y,x))
    return True

#Attach the click event to the window
cv.setMouseCallback("Thermal Camera", clickEvent)

#Main loop
frameC = 0
while True:
    #Camera read
    if cam.isOpened():
        ret, frame = cam.read()
        frameTime = time.time() #Here to prevent any issues with the time taken to process the frame
        if ret:
            frameC += 1
            frame = frame.flatten()
            frame = frame.reshape((resY*fNum,resX,-1)) #Leaving the last dimension as 2 for the 2 values of each pixel
            imdata, thdata = np.array_split(frame, fNum) #Splitting the frame into the contrasted image and the temperature matrix
            #Processing the temperature matrix
            tMatrix = np.zeros((resY,resX))
            #Smoothing the temperature matrix, if required
            if expToggle and smoothed:
                smoothed = False #Turn off smoothed data for experimental data
            tMatrix = calculate_temperature_matrix(thdata, resY, resX, smoothed, smoothOrder)
            # Find the maximum, minimum and mean temperature values and their coordinates
            if tempStats:
                max_temp = np.max(tMatrix)
                min_temp = np.min(tMatrix)
                max_temp_coords = np.unravel_index(np.argmax(tMatrix), tMatrix.shape)
                min_temp_coords = np.unravel_index(np.argmin(tMatrix), tMatrix.shape)
                averageTemp = np.mean(tMatrix)
            else:
                max_temp = 0
                min_temp = 0
                max_temp_coords = (0,0)
                min_temp_coords = (0,0)
                averageTemp = 0
            
            #Start processing the image data
            bgr = cv.cvtColor(imdata, cv.COLOR_YUV2BGR_YUYV) #Convert the real image to RGB
            #Contrast
            bgr = cv.convertScaleAbs(bgr, alpha=alpha)
            #Resize
            if interp == "cubic":
                bgr = cv.resize(bgr, (newX,newY), interpolation=cv.INTER_CUBIC)
            elif interp == "linear":
                bgr = cv.resize(bgr, (newX,newY), interpolation=cv.INTER_LINEAR)
            elif interp == "nearest":
                bgr = cv.resize(bgr, (newX,newY), interpolation=cv.INTER_NEAREST)
            #Blur
            if bRad > 0:
                if bRad%2 == 0:
                    bRad += 1
                bgr = cv.GaussianBlur(bgr, (bRad,bRad), 0)
            #Apply colormap
            if colormap == 0:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_JET)
                cmapText = "Jet"
            elif colormap == 1:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_HOT)
                cmapText = "Hot"
            elif colormap == 2:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_PARULA)
                cmapText = "Parula"
            elif colormap == 3:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_MAGMA)
                cmapText = "Magma"
            elif colormap == 4:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_INFERNO)
                cmapText = "Inferno"
            elif colormap == 5:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_PLASMA)
                cmapText = "Plasma"
            elif colormap == 6:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_BONE)
                cmapText = "Bone"
            elif colormap == 7:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_SPRING)
                cmapText = "Spring"
            elif colormap == 8:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_AUTUMN)
                cmapText = "Autumn"
            elif colormap == 9:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_VIRIDIS)
                cmapText = "Viridis"
            elif colormap == 10:
                heatmap = cv.applyColorMap(bgr, cv.COLORMAP_RAINBOW)
                heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
                cmapText = "Inv Rainbow"
            #Draw data
            for x in trackedPixels:
                t = tMatrix[x[0]//scale,x[1]//scale]
                drawData(x, t, heatmap)
            #Display data
            if hud:
                cv.putText(heatmap, f"Max Temp: {max_temp} K at {max_temp_coords}", (10, 20), font, 0.5, (0, 255, 255), 1, cv.LINE_AA)
                cv.putText(heatmap, f"Min Temp: {min_temp} K at {min_temp_coords}", (10, 40), font, 0.5, (0, 255, 255), 1, cv.LINE_AA)
                cv.putText(heatmap, f"Average Temp: {averageTemp: .2f} K", (10, 60), font, 0.5, (0, 255, 255), 1, cv.LINE_AA)
                cv.putText(heatmap, f"Colormap: {cmapText}", (10, 80), font, 0.5, (0, 255, 255), 1, cv.LINE_AA)
            #Display floating maximum and minimum temperature values, conditionally
            if tempStats:
                #cv.putText(heatmap, f"Max Temp: {max_temp} K", (10, 100), font, 0.5, (0, 255, 255), 1, cv.LINE_AA)
                #cv.putText(heatmap, f"Min Temp: {min_temp} K", (10, 120), font, 0.5, (0, 255, 255), 1, cv.LINE_AA)
                if max_temp > averageTemp + thres:
                    mrow = max_temp_coords[0]
                    mcol = max_temp_coords[1]
                    cv.circle(heatmap, (mcol*scale, mrow*scale), 5, (0,0,0), 2)
                    cv.circle(heatmap, (mcol*scale, mrow*scale), 5, (0,0,255), -1)
                    cv.putText(heatmap,str(max_temp)+' K', ((mcol*scale)+10, (mrow*scale)+5),\
                    font, 0.45,(0,0,0), 2, cv.LINE_AA)
                    cv.putText(heatmap,str(max_temp)+' K', ((mcol*scale)+10, (mrow*scale)+5),\
                    font, 0.45,(0, 255, 255), 1, cv.LINE_AA)
                if min_temp < averageTemp - thres:
                    mrow = min_temp_coords[0]
                    mcol = min_temp_coords[1]
                    cv.circle(heatmap, (mcol*scale, mrow*scale), 5, (0,0,0), 2)
                    cv.circle(heatmap, (mcol*scale, mrow*scale), 5, (0,0,255), -1)
                    cv.putText(heatmap,str(min_temp)+' K', ((mcol*scale)+10, (mrow*scale)+5),\
                    font, 0.45,(0,0,0), 2, cv.LINE_AA)
                    cv.putText(heatmap,str(min_temp)+' K', ((mcol*scale)+10, (mrow*scale)+5),\
                    font, 0.45,(0, 255, 255), 1, cv.LINE_AA)
            #Show the image
            if frameC%dispInt == 0: #Display every dispInt frames
                cv.imshow("Thermal Camera", heatmap)
            else:
                pass
        else:
            print("Error reading frame")
            break #Default behavior, as such exp data is compromised anyways
    elif not cam.isOpened():
        print("Error establishing connection")
        break #Exit the loop if the camera is not connected - default behavior, as such exp data is compromised anyways
    #Key press handling
    keyPress = cv.waitKey(1)
    if keyPress == 27: #Esc key
        cam.release()
        cv.destroyAllWindows()
        break
    elif keyPress == ord('s'):
        snap(heatmap,tMatrix)
    elif keyPress == ord('c'):
        #Clear all tracked pixels
        trackedPixels = []
    elif keyPress == ord('h'):
        #Toggle HUD
        hud = not hud
    elif keyPress == ord('m'):
        #Toggle temperature statistics display
        tempStats = not tempStats
    elif keyPress == ord('r'):
        #Reset tracked pixels and experimental data
        trackedPixels = []
        experimentalData = []
    elif keyPress == ord('q'):
        cam.release()
        cv.destroyAllWindows()
        break
    elif keyPress == ord('p'):
        #Print tracked pixels
        print(trackedPixels)
    elif keyPress == ord('t'):
        #Print experimental data - for debugging
        print(experimentalData)
    elif keyPress == ord('a'):
        #Toggle recording all temperature data
        recordAll = not recordAll
    elif keyPress == ord('x'):
        #Toggle smoothed temperature data
        smoothed = not smoothed
    elif keyPress == ord('1'):
        #Increase contrast
        alpha += 0.1
        alpha = round(alpha,1) #Round off to 1 decimal place
    elif keyPress == ord('2'):
        #Decrease contrast
        alpha -= 0.1
        alpha = round(alpha,1) #Round off to 1 decimal place
    elif keyPress == ord('3'):
        #Increase blur radius
        bRad += 2
    elif keyPress == ord('4'):
        #Decrease blur radius
        bRad -= 2
    elif keyPress == ord('5'):
        #Increase threshold for max and min temp display
        thres += 1
    elif keyPress == ord('6'):
        #Decrease threshold for max and min temp display
        thres -= 1
        if thres < 0:
            thres = 0
    elif keyPress == ord('7'):
        #Cycle through interpolation methods
        if interp == "cubic":
            interp = "linear"
        elif interp == "linear":
            interp = "nearest"
        elif interp == "nearest":
            interp = "cubic"
    elif keyPress == ord('8'):
        #Cycle through colormaps
        colormap += 1
        if colormap == 11: #11 is the number of colormaps
            colormap = 0
    elif keyPress == ord('9'):
        #Increase smoothing order
        smoothOrder += 1
    elif keyPress == ord('0'):
        #Decrease smoothing order
        smoothOrder -= 1
    elif keyPress == ord('e'):
        #Toggle experimental data recording
        time.sleep(0.1) #To prevent accidental double presses
        if expToggle:
            print("Experimental data recording stopped")
            startTime = 0 #Reset the start time
            expEndTimes.append(time.time()) #Record the end time
            experimentalData.append(data) #Record the data
            data = [] #Reset the data list
        else:
            print("Experimental data recording started")
            smoothed = False #Turn off smoothed data for experimental data
            noExp += 1 #Increment the number of experimental datasets
            data = [] #Reset the data list
            startTime = time.time() #Start time of the experiment
            expStartTimes.append(startTime) #Record the start time
        expToggle = not expToggle
        continue #Skip the rest of the loop - so as to not record the smooth data
    
    #Experiment handling
    #Please note the format of the data recorded here - different modes have different formats
    if expToggle:
        if recordAll:
            data.append((tMatrix,frameTime-startTime)) #Record all data
        else:
            d = (tMatrix[i[0]//scale,i[1]//scale] for i in trackedPixels) #Record only the selected pixels
            data.append((d,frameTime-startTime)) #Record the data

#Account for the case where the experiment was not stopped
if expToggle:
    expEndTimes.append(time.time()) #Record the end time if the experiment was not stopped
    experimentalData.append(list(data)) #Record the data if the experiment was not stopped

#Save the experimental data
if noExp > 0:
    #We have to deal with two cases here - one where the data is recorded for all pixels, and one where it is recorded for only selected pixels
    #The format of the data is different in each case
    for i in range(noExp):
        d = experimentalData[i]
        if type(d[0]) != tuple:
            #Data is recorded for all pixels
            #Not saving in CSV format as it is not very useful
            np.save(f"expData{expStartTimes[i]: .0f}-{expEndTimes[i]: .0f}_{i}.npy",np.asarray(d))
        else:
            #Data is recorded for only selected pixels
            with open(f"expData{round(expStartTimes[i])}-{round(expEndTimes[i])}_{i}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["Coordinates"]
                for j in range(len(d)):
                    headers.append(f"{j}")
                writer.writerow(headers)
                for k in range(len(trackedPixels)):
                    row = [trackedPixels[k]]
                    for j in range(len(d)):
                        data_list = list(d[j][0])
                        row.append(data_list[k])
                    writer.writerow(row)
#End of main loop
print("End of program")
print(f"Number of experimental datasets: {noExp}")
print("Please check the experimental data for any issues")
print("Please check the calibration data for any issues")
print("Created by: Diptanuj Sarkar (Jan 2025)")
print("Please report any issues to: diptanuj.ds@gmail.com (Or on the GitHub repository)")
print("Thank you for using this program")