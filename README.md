# topdon-tc001-python
My attempt at reverse-engineering(?) and calibrating the Topdon TC001 Thermal Imaging camera, so that I do not have to rely on third-party calibration and software.

## process.ipynb
I am in the process of understanding how the camera works, and I want to document how I arrive at my conclusions. To this end, I have put in the important steps in the process of creating software for this camera in this Jupyter Notebook (Python).
The dependencies are all noted in the notebook, and you can follow along on your system.

## camera.py
UPDATE: 23-01-2025

`camera.py` is a general-purpose software script designed for extracting useful calibrated data with error estimation from IR Thermal Cameras, specifically the Topdon TC001. The script follows a process to fetch data from the camera, reshape it, and calculate a temperature matrix based on a linear or quadratic fit of the pixel values versus calibrated temperature from a file. The temperature matrix is then used for displaying heatmaps, taking experimental measurements, plotting, etc.

### Key Bindings
- **Esc**: Exit the program
- **s**: Take a snapshot
- **c**: Clear all tracked pixels
- **h**: Toggle HUD (Heads-Up Display)
- **m**: Toggle temperature statistics display
- **r**: Reset tracked pixels and experimental data
- **q**: Quit the program
- **p**: Print tracked pixels
- **t**: Print experimental data
- **a**: Toggle recording all temperature data
- **x**: Toggle smoothed temperature data
- **1**: Increase contrast
- **2**: Decrease contrast
- **3**: Increase blur radius
- **4**: Decrease blur radius
- **5**: Increase threshold for max and min temperature display
- **6**: Decrease threshold for max and min temperature display
- **7**: Cycle through interpolation methods
- **8**: Cycle through colormaps
- **9**: Increase smoothing order
- **0**: Decrease smoothing order
- **e**: Toggle experimental data recording

### Setting Parameters
- **fitOrder**: Order of the fit for temperature calculation (1 = Linear, 2 = Quadratic)
- **resX**: Resolution of the camera in the x-direction (in pixels)
- **resY**: Resolution of the camera in the y-direction (in pixels)
- **fNum**: Number of frames returned by the camera (TC001 returns 2 frames in one go, a contrasted image and a temperature matrix)
- **interp**: Interpolation algorithm for image display (Options: "nearest", "linear", "cubic")
- **recordAll**: Toggle between recording all temperature data or just the selected pixels
- **plot**: Generate plots for the temperature data (True/False)
- **calibFile**: Filename of the calibration data
- **kelvin**: Toggle if the calibration data is based on the Kelvin scale (True/False)
- **dispInt**: Display interval (e.g., display every 5 frames)
- **dev**: USB Camera number (generally 0 if the thermal camera is the only USB camera connected)
- **smoothOrder**: Smoothing order for the temperature matrix (1 for 3x3, 2 for 5x5, 3 for 7x7, etc.)
- **smoothed**: Switch between displaying smoothed and raw temperature data (True/False)
- **tempStats**: Switch on/off maximum, minimum, and average temperature display (True/False)
- **scale**: Scale of the displayed image
- **alpha**: Controls contrast
- **colormap**: Colormap to be applied (0 = Jet, 1 = Hot, 2 = Parula, 3 = Magma, 4 = Inferno, 5 = Plasma, 6 = Bone, 7 = Spring, 8 = Autumn, 9 = Viridis, 10 = Inv Rainbow)
- **bRad**: Blur radius
- **thres**: Threshold for max and min temperature display
- **hud**: Heads-up display (True/False)

### Usage
1. Ensure the camera is connected and powered on.
2. Run the script.
3. Use the key bindings to interact with the program.
4. The script will display the thermal camera feed with various overlays and options based on the settings and key bindings.

### Notes
- Calibration data should be in the form of a NumPy array with dimensions (Y, X, 6) for a quadratic fit and (Y, X, 3) for a linear fit.
- The first 2 dimensions are the resolution of the camera, and the 3rd dimension is the coefficients of the fit.
- The coefficients are in the order: a, b, c, d, e, f for a quadratic fit and a, b, c for a linear fit.
- Ensure that the calibration data is in the same folder as the script or specify the path in the script.
- The script is designed to collect raw data from the camera, with smoothing, contrast adjustment, etc., done post-facto.

## generateDefaultCalib.py
Does nothing much. Creates a calibration data file called `calib.npy` that contains linear fit coefficients for the standard computation of temperature (in Celcius), given by:

$$
T = \frac{1}{64} \cdot \text{Hi} + \frac{256}{64} \text{Lo} - 273.15
$$

Feel free to use it and generate a workable `calib.npy` file if you do not have any of your own

## To-Do
- [ ] Collect Calibration data
- [ ] Regress data on multiple models
- [-] MAIN: Create software for using the camera, primarily for scientific purposes.
  - [-] Heatmaps
  - [-] Multiple (unlimited) pixel data monitoring
  - [-] Global scene temperature mean/median/max/min/etc. Possible floating display.
  - [ ] Temperature filtering based on range.
  - [-] Generating collected data with appropriate error estimates in .csv format.
  - [ ] Video recording.
- [ ] Possibly decoding USB communication with the camera (?)
