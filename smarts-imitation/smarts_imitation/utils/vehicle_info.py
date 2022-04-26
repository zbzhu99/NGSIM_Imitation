from collections import namedtuple

VehicleInfo = namedtuple(
    "VehicleInfo",
    ["vehicle_id", "start_time", "end_time", "scenario_name", "traffic_name"],
)

VehiclePosition = namedtuple(
    "VehiclePosition", ["position_x", "position_y", "sim_time"]
)
