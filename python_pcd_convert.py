import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d

BAG_FILE = "2025-02-06-19-09-18.bag"  # Replace with your .bag file path
TOPIC_NAME = "/velodyne_points"  # Replace with the correct topic name
OUTPUT_PCD = "output.pcd"  # Output PCD file name

def extract_pcd_from_bag(bag_file, topic_name, output_pcd):
    bag = rosbag.Bag(bag_file, "r")
    cloud_data = None

    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if topic == topic_name:
            print(f"Extracting point cloud from {t}")
            points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

            if points.shape[0] > 0:
                cloud_data = points
                break  # Extract only the first point cloud

    bag.close()

    if cloud_data is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_data)
        o3d.io.write_point_cloud(output_pcd, pcd)
        print(f"Saved PCD file: {output_pcd}")
    else:
        print("No point cloud data found!")

# Run extraction
extract_pcd_from_bag(BAG_FILE, TOPIC_NAME, OUTPUT_PCD)
