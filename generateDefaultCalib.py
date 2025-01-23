import numpy as np

# Create the numpy array with the specified shape and values
calib_array = np.zeros((192, 256, 3))
calib_array[:, :, 0] = 1 / 64
calib_array[:, :, 1] = 256 / 64
calib_array[:, :, 2] = -273.15

# Save the numpy array to a file
np.save('calib.npy', calib_array)