'''
ITC503S.py
This script is designed to interface with the ITC503S temperature controller using the PyVISA library.
It allows the user to select an instrument from the available resources, set up the instrument with specific parameters,
and query the instrument for temperature readings.
The script includes functions to initialize the instrument and to query it for temperature data.

Essentially, this runs a simple sweep program for the ITC503S temperature controller, from 300K to 310K. (Safe range)
It gives you a plot of the temperature, heater voltage, and set point over time.

Author: Diptanuj Sarkar (CasioWave)
'''
import pyvisa
import time
import numpy as np
import matplotlib.pyplot as plt

rm = pyvisa.ResourceManager()

resList = list(rm.list_resources())
print("Available resources: ",resList)

# Select the instrument
inst_str = resList[int(input("Select the instrument number: "))]

# Setup function for instrument
def setup_inst(inst_str, term_write, term_read, timeout, baudrate, initialize):
    '''
    Setup the instrument with the given parameters.
    Parameters:
    inst_str (str): The instrument string.
    term_write (str): The termination character for writing.
    term_read (str): The termination character for reading.
    timeout (int): The timeout for the instrument.
    baudrate (int): The baudrate for the instrument.
    initialize (list of str): The initialization commands for the instrument. 
    '''
    inst = rm.open_resource(inst_str)
    inst.read_termination = term_read
    inst.write_termination = term_write
    inst.timeout = timeout
    inst.baud_rate = baudrate
    for cmd in initialize:
        inst.write(cmd)
        time.sleep(0.1)
    print(f"Initialized with command: {initialize}")
    print(f"Instrument {inst_str} setup complete.")
    return inst

# Initialize the instrument
inst = setup_inst(
    inst_str,
    term_write='\r',
    term_read='\r',
    timeout=2000,
    baudrate=9600,
    initialize=[
        'C3',  # Set to remote mode
        'Q0',  # Set to default termination char mode
        'A1',  # Set heater to Auto mode
        'T300'  # Set temperature to 300K
    ]
)

# Function to read temperature
def query_itc503S(query,startChar='R',inst=inst):
    '''
    Query the ITC503S instrument for temperature.
    Parameters:
    query (str): The query string to send to the instrument.
    startChar (str): The character that indicates the start of the response.
    inst (pyvisa.Resource): The instrument resource.
    Returns:
    float: The response from the instrument as a float.
    '''
    inst.write(query)
    ret = ""
    while True:
        c = inst.read_bytes(1).decode('utf-8')
        if c == startChar:
            #print("Started reading temperature")
            while True:
                c = inst.read_bytes(1).decode('utf-8')
                if c == '\r':
                    break
                ret += c
            break
        else:
            #print("Waiting for 'R' character, got: ", c)
            pass
    return float(ret)

#Function to check if the target has equilibriated at set temperature
def check_equilibrium(target, tempHistory, tolerance=0.1, hist=30, inst=inst):
    '''
    Check if the target temperature has been reached within the specified tolerance.
    Parameters:
    target (float): The target temperature.
    tempHistory (list of 2-tuples): The history of temperature readings.
    tolerance (float): The tolerance for equilibrium.
    hist (int): Number of seconds from the history to check.
    inst (pyvisa.Resource): The instrument resource.
    Returns:
    bool: True if the target temperature is reached, False otherwise.
    '''
    #Calculate the variance of the last 'hist' seconds
    now = time.time()
    for i in range(len(tempHistory)):
        if abs(now-tempHistory[i][1]) > hist:
            break
        else:
            pass
    dat = tempHistory[i:]
    #Calculate the variance
    var = np.array([(target-x[0])**2 for x in dat])
    var = np.mean(var)
    #Check if the variance is within the tolerance
    if var < tolerance**2:
        #print("Equilibrium reached")
        return True
    else:
        #print("Equilibrium not reached")
        return False

#Sweep program
temps = [304,305,306]

dataTemp = []
dataSet = []
dataVolt = []

start = time.time()

def diffTime(start=start):
    '''
    Calculate the elapsed time since the start time.
    Parameters:
    start (float): The start time.
    Returns:
    float: The elapsed time in seconds.
    '''
    return time.time() - start

for i in temps:
    inst.write(f'T{i}')
    time.sleep(0.1)
    print(f"Set temperature to {i}K")
    while True:
        try:
            # Read the temperature, set point, and heater voltage
            dataTemp.append((query_itc503S('R1'),diffTime()))
            dataSet.append((query_itc503S('R0'),diffTime()))
            dataVolt.append((query_itc503S('R6'),diffTime()))
            if check_equilibrium(i, dataTemp, hist=30):
                print(f"Temperature: {dataTemp[-1][0]}K, Set Point: {dataSet[-1][0]}K, Heater Voltage: {dataVolt[-1][0]}V")
                print(f"Reached set point {i}K")
                break
        except:
            print("Error reading data, retrying...")
            time.sleep(0.1)

# Convert data to numpy arrays
dataTemp = np.array(dataTemp)
dataSet = np.array(dataSet)
dataVolt = np.array(dataVolt)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(dataTemp[:, 1], dataTemp[:, 0], label='Temperature (K)', color='blue')
plt.ylabel('Temperature (K)')
plt.title('Temperature, Set Point, and Heater Voltage over Time')
plt.grid()
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(dataSet[:, 1], dataSet[:, 0], label='Set Point (K)', color='orange')
plt.ylabel('Set Point (K)')
plt.grid()
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(dataVolt[:, 1], dataVolt[:, 0], label='Heater Voltage (V)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Heater Voltage (V)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Close the instrument
inst.close()
print("Instrument closed.")

# End of script
# Note: This script is designed to be run in an environment where the ITC503S temperature controller is connected.
# Ensure that the PyVISA library is installed and the instrument is connected properly.
# The script is intended for educational and research purposes. Use it at your own risk.
# Make sure to handle exceptions and errors as needed.