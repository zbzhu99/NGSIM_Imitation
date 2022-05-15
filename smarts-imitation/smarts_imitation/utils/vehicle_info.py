from collections import namedtuple

""" VehicleInfo
Args:
    vehicle_id (Required): Vehicle id of the vehicle.
    start_time (Optional): Specify the start_time of the vehicle (can not be the 
        "first-seen-time"). Default None.
    end_time (Optional): Specify the end_time of the vehicle (can not be the 
        "last-seen-time"). Default None.
    scenario_name (Required): Scenario name of the vehicle.
    traffic_name (Required): Traffic name of the vehicle.
    ttc (Optional): Time to collision of the vehicle, only used for cut-in cases.
"""
VehicleInfo = namedtuple(
    "VehicleInfo",
    ["vehicle_id", "start_time", "end_time", "scenario_name", "traffic_name", "ttc"],
)

VehiclePosition = namedtuple(
    "VehiclePosition", ["position_x", "position_y", "sim_time"]
)
