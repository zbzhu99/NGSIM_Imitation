from collections import namedtuple

VehicleWithTime = namedtuple(
    "VehicleWithTime", ["vehicle_id", "start_time", "end_time", "traffic_name"]
)
