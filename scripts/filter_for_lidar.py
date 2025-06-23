#!/usr/bin/env python3
import numpy as np
from typing import List, Optional
from sensor_msgs.msg import LaserScan
from scipy import signal

import rospy

class DynamicObstacleFilter:
    def __init__(self):
        rospy.init_node('range_filter')


        self.previous_distance = 0.0

        self.filtered_pub = rospy.Publisher("scan_filtered", LaserScan, queue_size=10)
        rospy.Subscriber("scan", LaserScan, self.laser_data_handler)

    def laser_data_handler(self, data: LaserScan) -> None:
        processed_ranges = self.process_ranges(data)

        filtered_msg = LaserScan()
        filtered_msg.header = data.header
        filtered_msg.angle_min = data.angle_min
        filtered_msg.angle_max = data.angle_max
        filtered_msg.angle_increment = data.angle_increment
        filtered_msg.time_increment = data.time_increment
        filtered_msg.scan_time = data.scan_time
        filtered_msg.range_min = data.range_min
        filtered_msg.range_max = data.range_max
        filtered_msg.ranges = processed_ranges

        self.filtered_pub.publish(filtered_msg)

    def process_ranges(self, scan_data: LaserScan) -> List[float]:
        if not scan_data.ranges:
            return []


        raw_distances = np.array(scan_data.ranges, dtype=np.float32)

        raw_distances[~np.isfinite(raw_distances)] = scan_data.range_max

        np.clip(raw_distances, scan_data.range_min, scan_data.range_max, out=raw_distances)

        window_size = 5
        smoothed = np.convolve(raw_distances, np.ones(window_size) / window_size, mode='same')

        cleaned = signal.medfilt(smoothed, kernel_size=window_size)

        np.clip(cleaned, scan_data.range_min, scan_data.range_max, out=cleaned)

        return cleaned.tolist()

def run_node():
    filter_node = DynamicObstacleFilter()
    rospy.spin()

if __name__ == '__main__':
    run_node()