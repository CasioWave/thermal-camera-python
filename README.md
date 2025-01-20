# topdon-tc001-python
My attempt at reverse-engineering(?) and calibrating the Topdon TC001 Thermal Imaging camera, so that I do not have to rely on third-party calibration and software.

## process.ipynb
I am in the process of understanding how the camera works, and I want to document how I arrive at my conclusions. To this end, I have put in the important steps in the process of creating software for this camera in this Jupyter Notebook (Python).
The dependencies are all noted in the notebook, and you can follow along on your system.

## main.py
File in development. The software is not intended to have many bells and whistles (You can find the planned features below) - I just need it to collect data in experiments.

## To-Do
- [ ] Collect Calibration data
- [ ] Regress data on multiple models
- [ ] MAIN: Create software for using the camera, primarily for scientific purposes.
  - [ ] Heatmaps
  - [ ] Multiple (unlimited) pixel data monitoring
  - [ ] Global scene temperature mean/median/max/min/etc. Possible floating display.
  - [ ] Temperature filtering based on range.
  - [ ] Generating collected data with appropriate error estimates in .csv format.
  - [ ] Video recording.
- [ ] Possibly decoding USB communication with the camera (?)
