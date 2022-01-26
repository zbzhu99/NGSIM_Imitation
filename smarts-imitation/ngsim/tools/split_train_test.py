import math
import argparse
import pandas as pd
import numpy as np
from os import path as osp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=990214,
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    df = pd.read_csv(
        args.data_path,
        sep=r"\s+",
        header=None,
        names=(
            "vehicle_id",
            "frame_id",  # 1 frame per .1s
            "total_frames",
            "global_time",  # msecs
            # front center in feet from left lane edge
            "position_x",
            # front center in feet from entry edge
            "position_y",
            "global_x",  # front center in feet
            "global_y",  # front center in feet
            "length",  # feet
            "width",  # feet
            "type",  # 1 = motorcycle, 2 = auto, 3 = truck
            "speed",  # feet / sec
            "acceleration",  # feet / sec^2
            "lane_id",  # lower is further left
            "preceding_vehicle_id",
            "following_vehicle_id",
            "spacing",  # feet
            "headway",  # secs
        ),
    )

    vehicle_ids = df["vehicle_id"].unique()
    vehicle_num = len(vehicle_ids)
    train_ids = np.random.choice(
        vehicle_ids, math.floor(vehicle_num * 0.8), replace=False
    )
    test_ids = np.setdiff1d(vehicle_ids, train_ids)

    train_df = df[df["vehicle_id"].isin(train_ids)]
    test_df = df[df["vehicle_id"].isin(test_ids)]

    train_df.to_csv(
        osp.join(osp.dirname(osp.abspath(args.data_path)), "train.txt"),
        sep=" ",
        header=None,
    )

    test_df.to_csv(
        osp.join(osp.dirname(osp.abspath(args.data_path)), "test.txt"),
        sep=" ",
        header=None,
    )
