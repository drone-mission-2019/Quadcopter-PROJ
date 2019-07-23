try:
    import vrep 
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')  

import time 

print('Program Started')
vrep.simxFinish(-1) #close all opened connections
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID != -1:
    print('Connected to Remote API Server...')
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
    res, planeHandle = vrep.simxGetObjectHandle(clientID, 'Quadricopter#', vrep.simx_opmode_blocking)
    
    #res, retInt, retFloat, retString, retBuffer = vrep.simxCallScriptFunction(clientID, 'misson_landing', )
    time.sleep(10)
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
    print('Program ended')
