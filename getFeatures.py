import laspy
import sys
import numpy as np
from scipy.spatial import cKDTree
import time
import send_email
import calculateFeatures
import resource

init_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Initial memory usage: {init_memory_usage / (1024 * 1024)} mb")

start = time.time()
input_path = '../working/classification/multiscale/classified_sample.las'
input_path = '../working/isolated_features/car_training.las'
output_file = '../working/files/geometric_features'
radius = sys.argv[1] #search radius


point_cloud = laspy.read(input_path)
#get info
points = np.vstack((point_cloud.x,
                    point_cloud.y,
                    point_cloud.z,
                    point_cloud['normal z'],
                    point_cloud.classification,
                    point_cloud.red,
                    point_cloud.green,
                    point_cloud.blue)).transpose()
features = calculateFeatures.calculateGeometricFeatures(points,radius,save=True,output_file=output_file)

end = time.time()
duration = end-start
if duration/60 > 30:
    duration_parsed = str(round((end-start)/3600,3)) + 'hours'
else:
    duration_parsed = str(round((end-start)/60,3)) + 'minutes'

print_message=f'Calculations for {input_path} Radius: {radius} are done and saved in {output_file}.\n Time elapsed: {duration_parsed}.'
print(print_message)

#add mailme to CLI and get an email notification sent when scipt is done
try:
    #check if mailme command exists
    if len(sys.argv)>=2 and sys.argv[2]=='mailme':
            send_email.sendNotification(f'Geometric feature calculation finished. {print_message}')
except:
    print("Mail was not send due to API error")

max_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Maximum memory usage: {max_memory_usage / (1024 * 1024)} mb")

print(f"Delta memory usage: {(max_memory_usage - init_memory_usage)/(1024 * 1024)} mb")
