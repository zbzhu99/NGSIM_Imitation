from dataloader import DataLoader
import torch
from collections import namedtuple
import pickle
import utils

print("> Loading DataLoader")


class opt:
    debug = 0


dataloader = DataLoader(None, opt, "i80")

print("> Loading splits")
splits = torch.load(
    "/NAS2020/Workspaces/DRLGroup/zbzhu/lfo-ppuu/pytorch-PPUU/traffic-data/state-action-cost/data_i80_v0/splits.pth"
)

black_list = [169]

for split in splits:
    if split == "valid_indx":
        continue
    data_dict = dict()
    print(f"> Building {split}")
    for idx in splits[split]:
        car_path = dataloader.ids[idx]
        states = dataloader.states[idx]
        timeslot, car_id = utils.parse_car_path(car_path)
        if car_id in black_list:
            continue
        data_dict[idx] = timeslot, car_id
    print(f"> Pickling {split}, length of {len(data_dict)}")
    with open(f"{split}_final.pkl", "wb") as f:
        pickle.dump(data_dict, f)
