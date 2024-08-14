import numpy as np
import plyfile

import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from pyntcloud import PyntCloud
import pickle


# Load PLY file
with open('Cs_POINTS.pkl', 'rb') as f:
    points = pickle.load(f)

point_cloud_raw = points  # This is a N*3 matrix of xyz coordinates
point_cloud_raw = DataFrame(point_cloud_raw[:,0:3])  # Select the first two elements from each column
point_cloud_raw.columns = ['x', 'y', 'z']  # Assign column titles to the selected data
point_cloud_pynt = PyntCloud(point_cloud_raw)  # Store the data in the PyntCloud structure

point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # Instantiate

o3d.visualization.draw_geometries([point_cloud_o3d])  # Display the original point cloud
