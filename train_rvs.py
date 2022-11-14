# Add offline training code for Track-2 here.
# On completion of training, automatically save the trained model to `track2/submission` directory.

import argparse
import glob
import re
import numpy as np
import os
import pickle
import sys
import shutil
import yaml
from pathlib import Path
from PIL import Image
from typing import Any, Dict, Optional
import wandb
import random
import pickle
import lhb_d3rlpy
from lhb_d3rlpy.dataset import MDPDataset
from lhb_d3rlpy.preprocessing import MinMaxActionScaler
from lhb_d3rlpy.algos import CQL
from copy import deepcopy
from tqdm import tqdm

# 模型转储
import torch
import math

# designed obs and rew
from env_wrapper import EnvWrapper, SingleAgentWrapper

# To import submission folder
sys.path.insert(0, str(Path(__file__).parents[1]))

from submission.utility import (
    goal_region_reward,
    get_goal_layer,
    get_trans_coor,
)


def load_config(path: Path) -> Optional[Dict[str, Any]]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


class MyWaypoint:
    def __init__(self, x, y, heading):
        self.pos = np.array([x, y])
        self.heading = heading


def gen_my_wp_from_obs(raw_obs):
    return MyWaypoint(raw_obs.ego_vehicle_state.position[0],
                      raw_obs.ego_vehicle_state.position[1],
                      float(raw_obs.ego_vehicle_state.heading))


def gen_my_wp_from_path(wp):
    return MyWaypoint(wp.pos[0],
                      wp.pos[1],
                      float(wp.heading))


def gen_wp_path(vehicle_data, step, gap_threshold=0.3):
    data = list(vehicle_data.values())[step:]  # 这里step应该从0开始计数
    lane_index = data[0].ego_vehicle_state.lane_index
    try:
        current_path = data[0].waypoint_paths[lane_index]
    except:
        current_path = []
    path_wps = [gen_my_wp_from_path(wp) for wp in current_path]
    future_distance_travelled = ([obs.distance_travelled for obs in data[1:]] + [0])
    future_distance_travelled = [obs.distance_travelled for obs in data[1:]]
    track_wps = [gen_my_wp_from_obs(data[0])]
    acc_distance_travelled = 0
    for i, raw_obs in enumerate(data[1:]):
        wp = gen_my_wp_from_obs(raw_obs)
        acc_distance_travelled += raw_obs.distance_travelled
        if acc_distance_travelled >= gap_threshold:
            track_wps.append(wp)
            acc_distance_travelled = 0
    if len(track_wps) >= 5 + 1:
        return track_wps[1:6]
    elif len(path_wps) >= 6 and cal_wps_bias(current_path) < 0.1:
        return path_wps[1:6]
    else:
        return None


def cal_wps_bias(current_path):
    headings = []
    if len(headings) < 5:
        return 1e5
    for wp in current_path[:5]:
        headings.append(wp.heading)

    abs_bias = 0
    for i, heading in enumerate(headings[:-1]):
        next_heading = headings[i + 1]
        abs_bias += abs(next_heading - heading)
    return abs_bias


def train(input_path, output_suffix, conservative_weight):
    lhb_d3rlpy.seed(313)

    # Get config parameters.
    train_config = load_config(Path(__file__).absolute().parents[0] / "config.yaml")

    n_steps = train_config["n_steps"]
    n_steps_per_epoch = train_config["n_steps_per_epoch"]
    n_scenarios = train_config["n_scenarios"]
    n_vehicles = train_config["n_vehicles"]
    gpu = train_config["gpu"]

    scenarios = list()
    for scenario_name in os.listdir(input_path):
        scenarios.append(scenario_name)
    save_directory = Path(__file__).absolute().parents[0] / "d3rlpy_logs_{}".format(output_suffix)
    if not os.path.isdir(save_directory):
        index = 0
        os.mkdir(save_directory)
    else:
        index = len(os.listdir(save_directory))
    # on shoulder test
    # scenarios = ['a81d659e56d1d7b5']
    if n_scenarios == "max" or n_scenarios > len(scenarios):
        n_scenarios = len(scenarios)  # 2

    # init wandb
    wandb.init(project='mlp_cql_simple')

    wash_data = True

    # 读取SMARTS格式数据集，并将其转化为MDP Dataset
    cnt = 0
    obs = list()
    actions = list()
    rewards = list()
    terminals = list()
    episode_terminals = list()
    turning_files = []
    straight_files = []
    for sc_index, scenario in tqdm(enumerate(scenarios[0:n_scenarios])):
        if sc_index % 10 == 0:
            print(f"Processing scenario {scenario}.")
        vehicle_ids = list()

        scenario_path = Path(input_path) / scenario
        for filename in os.listdir(scenario_path):
            if filename.endswith(".pkl"):
                match = re.search("vehicle-(.*).pkl", filename)
                assert match is not None
                vehicle_id = match.group(1)
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)

        n_vehicles = len(vehicle_ids)
        for id in vehicle_ids[0:n_vehicles]:
            if sc_index % 20 == 0:
                print(f"Adding data for vehicle id {id} in scenario {scenario}.")
            with open(
                    scenario_path / (f"Agent-history-vehicle-{id}.pkl"),
                    "rb",
            ) as f:
                vehicle_data = pickle.load(f)
            image_names = list()

            for filename in os.listdir(scenario_path):
                if filename.endswith(f"-{id}.png"):
                    image_names.append(filename)

            image_names = sorted(image_names)

            goal_pos_x = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[0]
            goal_pos_y = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[1]

            threshold = 3
            if wash_data:
                dheadings = []
                raw_headings = []

                raw_headings = [obs.ego_vehicle_state.heading for obs in vehicle_data.values()]
                all_ts = [float(name.split("_Agent")[0]) for name in image_names]
                vehicle_ts = [obs.elapsed_sim_time for obs in vehicle_data.values()]
                assert len(all_ts) == len(vehicle_data) and np.all(np.array(vehicle_ts) == np.array(all_ts)), 'wrong ts'
                ts, error_ts, correct_headings = [], [], []

                for i, heading in enumerate(raw_headings):
                    is_outlier = True
                    neighbor_headings = []
                    if i > 0:
                        neighbor_headings += raw_headings[max(0, i - 3):i]
                    if i < len(raw_headings) - 1:
                        neighbor_headings += raw_headings[i + 1:i + 4]
                    for neighbor_heading in neighbor_headings:
                        if abs(heading - neighbor_heading) < 8 * math.pi / 180:  # 我们认为正常情况下不会有车辆在0.1s转弯8度
                            is_outlier = False
                            break
                    if is_outlier:
                        # error_ts.append(image_names)
                        error_ts.append((i + 1) / 10)  # 我们暂时假设，车辆的ts从0.1开始，以step=0.1不间断线性增长
                    else:
                        ts.append((i + 1) / 10)
                        correct_headings.append(heading)
                fixed_headings = np.interp(error_ts, ts, correct_headings)  # 线性插值，修复离群点
                for i, x in enumerate(error_ts):
                    ts.append(x)
                    correct_headings.append(fixed_headings[i])
                ts_heading_pairs = zip(ts, correct_headings)
                ts_heading_pairs = sorted(ts_heading_pairs, key=lambda outlier: outlier[0])
                last_heading = ts_heading_pairs[0][1]
                abs_dheadings = []
                for t, heading in ts_heading_pairs[1:]:
                    abs_dheadings.append(
                        min(abs(heading - last_heading), 2 * math.pi - abs(heading - last_heading)) * 180 / math.pi)
                    last_heading = heading

                # for i in range(len(image_names) - 1):
                #     sim_time = image_names[i].split("_Agent")[0]
                #     sim_time_next = image_names[i + 1].split("_Agent")[0]
                #
                #     # raw_obs for env_wrapper
                #     current_raw_obs = vehicle_data[float(sim_time)]
                #     next_raw_obs = vehicle_data[float(sim_time_next)]
                #     raw_headings.append(current_raw_obs.ego_vehicle_state.heading)
                #     if i == len(image_names) - 2:
                #         raw_headings.append(next_raw_obs.ego_vehicle_state.heading)

                if len(raw_headings) < 3:
                    continue

                # def abs_dheading(h1, h2):
                #     return min(abs(h1 - h2), abs(2 * math.pi - h1 + h2))
                #
                # def check(h1, h2):
                #     return abs_dheading(h1, h2) >= 10 * math.pi / 180

                # if check(raw_headings[0], raw_headings[1]) and check(raw_headings[0], raw_headings[2]):
                #     raw_headings[0] = 2 * raw_headings[1] - raw_headings[2]
                # for i, raw_h in enumerate(raw_headings[1:-1]):
                #     ind = i + 1
                #     last_raw_h = raw_headings[ind - 1]
                #     next_raw_h = raw_headings[ind + 1]
                #     if check(raw_h, last_raw_h) and check(raw_h, next_raw_h):
                #         raw_headings[ind] = (last_raw_h + next_raw_h) / 2
                #         print('smooth', last_raw_h, raw_h, next_raw_h)
                # if check(raw_headings[-1], raw_headings[-2]) and check(raw_headings[-1], raw_headings[-3]):
                #     raw_headings[-1] = 2 * raw_headings[-2] - raw_headings[-3]
                # for i in range(len(raw_headings) - 1):
                #     dheadings.append(raw_headings[i + 1] - raw_headings[i])

                if max(abs_dheadings) >= 0.5 and scenario not in turning_files:
                    turning_files.append(scenario)
    # if wash_data:
    #     for i, file_n in enumerate(turning_files):
    #         print(i, file_n)
    #         os.system(f"cp -r ../offline_datasets_1006/{file_n} ./change_or_turn_05run_fix/{file_n}")
    #     return
    last_len = 0
    print('start processing turning/changing lane data')
    for sc_index, scenario in tqdm(enumerate(turning_files)):  # 完成清洗，开始抽取数据，构造MDPdataset
        if sc_index % 20 == 0:
            print(f"Processing turning/changing lane scenario {scenario}.")
        vehicle_ids = list()

        scenario_path = Path(input_path) / scenario
        for filename in os.listdir(scenario_path):
            if filename.endswith(".pkl"):
                match = re.search("vehicle-(.*).pkl", filename)
                assert match is not None
                vehicle_id = match.group(1)
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)

        n_vehicles = len(vehicle_ids)
        for id in vehicle_ids[0:n_vehicles]:
            with open(
                    scenario_path / (f"Agent-history-vehicle-{id}.pkl"),
                    "rb",
            ) as f:
                vehicle_data = pickle.load(f)
            image_names = list()

            for filename in os.listdir(scenario_path):
                if filename.endswith(f"-{id}.png"):
                    image_names.append(filename)

            image_names = sorted(image_names)

            goal_pos_x = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[0]
            goal_pos_y = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[1]

            threshold = 3

            # 筛选出转弯/换道场景中位于转弯前后的数据
            raw_headings = [obs.ego_vehicle_state.heading for obs in vehicle_data.values()]
            all_ts = [float(name.split("_Agent")[0]) for name in image_names]
            vehicle_ts = [obs.elapsed_sim_time for obs in vehicle_data.values()]
            assert len(all_ts) == len(vehicle_data) and np.all(np.array(vehicle_ts) == np.array(all_ts)), 'wrong ts'
            ts, error_ts, correct_headings = [], [], []

            for i, heading in enumerate(raw_headings):
                is_outlier = True
                neighbor_headings = []
                if i > 0:
                    neighbor_headings += raw_headings[max(0, i - 3):i]
                if i < len(raw_headings) - 1:
                    neighbor_headings += raw_headings[i + 1:i + 4]
                for neighbor_heading in neighbor_headings:
                    if abs(heading - neighbor_heading) < 8 * math.pi / 180:  # 我们认为正常情况下不会有车辆在0.1s转弯8度
                        is_outlier = False
                        break
                if is_outlier:
                    # error_ts.append(image_names)
                    error_ts.append((i + 1) / 10)  # 我们暂时假设，车辆的ts从0.1开始，以step=0.1不间断线性增长
                else:
                    ts.append((i + 1) / 10)
                    correct_headings.append(heading)
            fixed_headings = np.interp(error_ts, ts, correct_headings)  # 线性插值，修复离群点
            for i, x in enumerate(error_ts):
                ts.append(x)
                correct_headings.append(fixed_headings[i])
            ts_heading_pairs = zip(ts, correct_headings)
            ts_heading_pairs = sorted(ts_heading_pairs, key=lambda outlier: outlier[0])
            last_heading = ts_heading_pairs[0][1]
            abs_dheadings = []
            for t, heading in ts_heading_pairs[1:]:
                abs_dheadings.append(
                    min(abs(heading - last_heading), 2 * math.pi - abs(heading - last_heading)) * 180 / math.pi)
                last_heading = heading
            abs_dheadings = ['None'] + abs_dheadings  # 加入占位符‘None’

            for i in range(0, len(image_names) - 1):  # i的含义是step cnt，从0开始
                with Image.open(scenario_path / image_names[i], "r") as image:
                    image.seek(0)
                    sim_time = image_names[i].split("_Agent")[0]
                    sim_time_next = image_names[i + 1].split("_Agent")[0]
                if max(abs_dheadings[max(1, i - 10):i + 10 + 1]) < 0.5:  # 仅使用转弯/并道前后10个step的action
                    continue

                # todo 确定observation，包括waypoint_obs和neighbor_obs
                fixed_waypoints = gen_wp_path(vehicle_data, step=i, gap_threshold=0.3)
                if fixed_waypoints is None:
                    continue  # 如果无法取得5个有效的wp，则抛弃当前step的数据
                # todo 在这算两种obs，记得做坐标系转换
                # ego_vehicle_state
                current_raw_obs = vehicle_data[float(sim_time)]
                state = current_raw_obs.ego_vehicle_state
                pos = state.position[:2]
                heading = float(state.heading)
                speed = state.speed
                lane_index = state.lane_index
                rotate_M = np.array([
                    [np.cos(heading), np.sin(heading)],
                    [-np.sin(heading), np.cos(heading)]]
                )  # heading是与y正半轴的夹角，这样转化是否会出现问题？

                ego_lane_positions = np.array([wp.pos for wp in fixed_waypoints])
                ego_lane_headings = np.array([float(wp.heading) for wp in fixed_waypoints])

                all_lane_rel_position = (
                        (ego_lane_positions.reshape(-1, 2) - pos.reshape(1, 2)) @ rotate_M.T).reshape(5, 2)
                all_lane_rel_heading = (ego_lane_headings - heading)
                all_lane_rel_heading[np.where(all_lane_rel_heading > np.pi)] -= np.pi  # 这是在干嘛
                all_lane_rel_heading[np.where(all_lane_rel_heading < -np.pi)] += np.pi

                EnvInfo_rel_pos_heading = np.zeros((1, 15))
                EnvInfo_speed_limit = np.zeros((1, 1))
                EnvInfo_bounding_box = np.zeros((1, 2))
                EnvInfo_lane_width = np.zeros((1, 1))
                EnvInfo_rel_pos_heading[0, :10] = all_lane_rel_position.reshape(
                    10, )  # todo 看一下测试环境的wp[0].heading是否始终是0
                EnvInfo_rel_pos_heading[0, 10:] = all_lane_rel_heading.reshape(5, )
                try:
                    speed_limit = vehicle_data[float(sim_time)].waypoint_paths[lane_index][0].speed_limit
                    # 有时waymo数据集的lane_index会out of range
                except:
                    speed_limit = vehicle_data[float(sim_time)].waypoint_paths[0][0].speed_limit
                EnvInfo_speed_limit[0, 0] = speed_limit
                EnvInfo_bounding_box[0, 0] = state.bounding_box.length
                EnvInfo_bounding_box[0, 1] = state.bounding_box.width

                EnvInfo = np.concatenate([
                    EnvInfo_rel_pos_heading,  # 15
                    EnvInfo_speed_limit,  # 1
                    EnvInfo_bounding_box,  # 2
                ], -1).astype(np.float32)

                # Neighbor Info
                on_road_neighbors = []
                for neighbor in current_raw_obs.neighborhood_vehicle_states:
                    if neighbor.lane_id != 'off_lane':
                        on_road_neighbors.append(neighbor)
                on_road_neighbors = on_road_neighbors[:50]
                neighbors_pos = np.zeros([50, 2])
                neighbors_bounding_box = np.zeros([50, 2])
                neighbors_speed = np.zeros(50)
                neighbors_heading = np.zeros([50])
                for nb_ind, neighbor in enumerate(on_road_neighbors):
                    neighbors_pos[nb_ind] = neighbor.position[:2]
                    neighbors_speed[nb_ind] = neighbor.speed
                    neighbors_heading[nb_ind] = neighbor.heading
                    neighbors_bounding_box[nb_ind, 0] = neighbor.bounding_box.length
                    neighbors_bounding_box[nb_ind, 1] = neighbor.bounding_box.width
                nb_num = len(on_road_neighbors)
                neighbors_rel_vel = np.zeros((50, 2))
                neighbors_rel_vel[:nb_num, 0] = -np.sin(neighbors_heading[:nb_num]) * neighbors_speed[:nb_num] + np.sin(
                    heading) * speed
                neighbors_rel_vel[:nb_num, 1] = np.cos(neighbors_heading[:nb_num]) * neighbors_speed[:nb_num] - np.cos(
                    heading) * speed

                nb_mask = np.all(neighbors_pos == 0, -1).astype(np.float32)

                neighbors_dist = np.sqrt(((neighbors_pos - pos) ** 2).sum(-1)) + nb_mask * 1e5
                st = np.argsort(neighbors_dist)[:5]

                NeighborInfo_rel_pos = ((neighbors_pos[st] - pos) @ rotate_M.T)
                NeighborInfo_rel_vel = ((neighbors_rel_vel[st]) @ rotate_M.T)
                NeighborInfo_rel_heading = (neighbors_heading - heading)[st].reshape(5, 1)
                NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading > np.pi)] -= np.pi
                NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading < -np.pi)] += np.pi
                NeighborInfo_boundingbox = neighbors_bounding_box[st]

                NeighborInfo = np.concatenate([
                    NeighborInfo_rel_pos,  # 2
                    NeighborInfo_rel_vel,  # 2
                    NeighborInfo_rel_heading,  # 1
                    NeighborInfo_boundingbox,  # 2
                ], -1).astype(np.float32)

                if len(on_road_neighbors) < 5:  # padding
                    NeighborInfo[len(on_road_neighbors):] = 0

                obs.append(np.concatenate([
                    NeighborInfo.reshape(-1, ),  # (25)
                    EnvInfo.reshape(-1, ),  # (15)
                ]))

                # 确定action：相对速度小于0.2->0，不小于0.2->1
                speed, heading, lane_index, lane_id = state.speed, state.heading, state.lane_index, state.lane_id
                try:
                    speed_limit = current_raw_obs.waypoint_paths[lane_index][0].speed_limit
                    # 有时waymo数据集的lane_index会out of range
                except:
                    speed_limit = current_raw_obs.waypoint_paths[0][0].speed_limit
                if speed / speed_limit >= 0.2:
                    action = 1  # moving
                else:
                    action = 0  # keeping still
                actions.append(action)
                # 确定terminal
                next_raw_obs = vehicle_data[float(sim_time_next)]
                events = next_raw_obs.events
                # 使用next_obs判定是否到达目的地，若到达则terminal
                next_position = vehicle_data[float(sim_time_next)].ego_vehicle_state.position
                goal_reward = goal_region_reward(
                    threshold,
                    goal_pos_x,
                    goal_pos_y,
                    next_position[0],
                    next_position[1],
                )  # 若到达终点则奖励+10，不确定是否要使用这一项
                if float(sim_time) > 8.7:
                    aaa = 1
                is_reached_goal = True if goal_reward > 0 else False
                if (not (events.off_road or events.reached_goal or events.reached_max_episode_steps or
                         is_reached_goal)) and len(events.collisions) == 0:
                    terminal = 0
                else:
                    terminal = 1
                terminals.append(terminal)
                # 确定reward：[distance_travelled, action_reward, goal_reward]
                dist_reward = next_raw_obs.distance_travelled
                action_reward = {
                    0: 0,
                    1: 1
                }[action]
                reward = [dist_reward, action_reward, goal_reward]

                rewards.append(reward)
                episode_terminals.append(0)
                if terminal == 1:
                    break
            episode_terminals[-1] = 1  # 在每个episode的最后一个step，将episode_terminals的最后一个元素修改为1
            print(f'ind {sc_index} scen {scenario} total_len {len(obs) - last_len} ter_sum {sum(episode_terminals)}')
            last_len = len(obs)
    aaa = 1

    # 计算直行轨迹的数量和长度
    turning_trajectory_num = len(turning_files)
    turning_trajectory_len = len(obs) // turning_trajectory_num  # 转弯/换道数据集中轨迹平均长度
    all_straight_files = []
    for sc_index, scenario in enumerate(scenarios[0:n_scenarios]):
        if scenario not in turning_files:
            all_straight_files.append(scenario)
    straight_trajectory_num = min(len(all_straight_files), turning_trajectory_num // 2)
    straight_trajectory_len = turning_trajectory_len

    seed = 20
    np.random.seed(seed)
    random.seed(seed)
    straight_idxs = np.random.choice(a=len(all_straight_files), size=straight_trajectory_num, replace=False,
                                     p=None).tolist()
    straight_files = [all_straight_files[idx] for idx in straight_idxs]  # 抽取直行scenario

    print('start processing straight data')
    last_len = len(obs)
    for sc_index, scenario in tqdm(enumerate(straight_files)):  # 开始抽取直行数据，构造MDPdataset
        if sc_index % 20 == 0:
            print(f"Processing straight scenario {scenario}.")
        vehicle_ids = list()

        scenario_path = Path(input_path) / scenario
        for filename in os.listdir(scenario_path):
            if filename.endswith(".pkl"):
                match = re.search("vehicle-(.*).pkl", filename)
                assert match is not None
                vehicle_id = match.group(1)
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)

        n_vehicles = len(vehicle_ids)
        for id in vehicle_ids[0:n_vehicles]:
            with open(
                    scenario_path / (f"Agent-history-vehicle-{id}.pkl"),
                    "rb",
            ) as f:
                vehicle_data = pickle.load(f)
            image_names = list()

            for filename in os.listdir(scenario_path):
                if filename.endswith(f"-{id}.png"):
                    image_names.append(filename)

            image_names = sorted(image_names)

            goal_pos_x = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[0]
            goal_pos_y = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[1]

            threshold = 3

            tmp_trajectory_len = min(len(vehicle_data) - 10, straight_trajectory_len)
            if tmp_trajectory_len < 10:  # 抛弃过短的直行轨迹
                continue

            # 修复轨迹中异常的heading值
            raw_headings = [obs.ego_vehicle_state.heading for obs in vehicle_data.values()]
            all_ts = [float(name.split("_Agent")[0]) for name in image_names]
            vehicle_ts = [obs.elapsed_sim_time for obs in vehicle_data.values()]
            assert len(all_ts) == len(vehicle_data) and np.all(np.array(vehicle_ts) == np.array(all_ts)), 'wrong ts'
            ts, error_ts, correct_headings = [], [], []

            for i, heading in enumerate(raw_headings):
                is_outlier = True
                neighbor_headings = []
                if i > 0:
                    neighbor_headings += raw_headings[max(0, i - 3):i]
                if i < len(raw_headings) - 1:
                    neighbor_headings += raw_headings[i + 1:i + 4]
                for neighbor_heading in neighbor_headings:
                    if abs(heading - neighbor_heading) < 8 * math.pi / 180:  # 我们认为正常情况下不会有车辆在0.1s转弯8度
                        is_outlier = False
                        break
                if is_outlier:
                    # error_ts.append(image_names)
                    error_ts.append((i + 1) / 10)  # 我们暂时假设，车辆的ts从0.1开始，以step=0.1不间断线性增长
                else:
                    ts.append((i + 1) / 10)
                    correct_headings.append(heading)
            fixed_headings = np.interp(error_ts, ts, correct_headings)  # 线性插值，修复离群点
            for i, x in enumerate(error_ts):
                ts.append(x)
                correct_headings.append(fixed_headings[i])
            ts_heading_pairs = zip(ts, correct_headings)
            ts_heading_pairs = sorted(ts_heading_pairs, key=lambda outlier: outlier[0])
            last_heading = ts_heading_pairs[0][1]
            abs_dheadings = []
            for t, heading in ts_heading_pairs[1:]:
                abs_dheadings.append(
                    min(abs(heading - last_heading), 2 * math.pi - abs(heading - last_heading)) * 180 / math.pi)
                last_heading = heading
            abs_dheadings = ['None'] + abs_dheadings  # 加入占位符‘None’

            trajectory_offset = random.randint(0, len(vehicle_data) - 10 - tmp_trajectory_len)
            # 轨迹第一个step的index;尽量不使用最后10个step的数据，因为可能没有足够的future step来拟合waypoint

            for i in range(0, len(image_names) - 1):  # i的含义是step cnt，从0开始
                with Image.open(scenario_path / image_names[i], "r") as image:
                    image.seek(0)
                    sim_time = image_names[i].split("_Agent")[0]
                    sim_time_next = image_names[i + 1].split("_Agent")[0]
                if i < trajectory_offset or i >= trajectory_offset + tmp_trajectory_len:  # 抽取tmp_trajectory_len个连续step的数据
                    continue

                # todo 确定observation，包括waypoint_obs和neighbor_obs
                fixed_waypoints = gen_wp_path(vehicle_data, step=i, gap_threshold=0.3)
                if fixed_waypoints is None:
                    continue  # 如果无法取得5个有效的wp，则抛弃当前step的数据
                # todo 在这算两种obs，记得做坐标系转换
                # ego_vehicle_state
                current_raw_obs = vehicle_data[float(sim_time)]
                state = current_raw_obs.ego_vehicle_state
                pos = state.position[:2]
                heading = float(state.heading)
                speed = state.speed
                lane_index = state.lane_index
                rotate_M = np.array([
                    [np.cos(heading), np.sin(heading)],
                    [-np.sin(heading), np.cos(heading)]]
                )  # heading是与y正半轴的夹角，这样转化是否会出现问题？

                ego_lane_positions = np.array([wp.pos for wp in fixed_waypoints])
                ego_lane_headings = np.array([float(wp.heading) for wp in fixed_waypoints])

                all_lane_rel_position = (
                        (ego_lane_positions.reshape(-1, 2) - pos.reshape(1, 2)) @ rotate_M.T).reshape(5, 2)
                all_lane_rel_heading = (ego_lane_headings - heading)
                all_lane_rel_heading[np.where(all_lane_rel_heading > np.pi)] -= np.pi  # 这是在干嘛
                all_lane_rel_heading[np.where(all_lane_rel_heading < -np.pi)] += np.pi

                EnvInfo_rel_pos_heading = np.zeros((1, 15))
                EnvInfo_speed_limit = np.zeros((1, 1))
                EnvInfo_bounding_box = np.zeros((1, 2))
                EnvInfo_lane_width = np.zeros((1, 1))
                EnvInfo_rel_pos_heading[0, :10] = all_lane_rel_position.reshape(
                    10, )  # todo 看一下测试环境的wp[0].heading是否始终是0
                EnvInfo_rel_pos_heading[0, 10:] = all_lane_rel_heading.reshape(5, )
                try:
                    speed_limit = vehicle_data[float(sim_time)].waypoint_paths[lane_index][0].speed_limit
                    # 有时waymo数据集的lane_index会out of range
                except:
                    speed_limit = vehicle_data[float(sim_time)].waypoint_paths[0][0].speed_limit
                EnvInfo_speed_limit[0, 0] = speed_limit
                EnvInfo_bounding_box[0, 0] = state.bounding_box.length
                EnvInfo_bounding_box[0, 1] = state.bounding_box.width

                EnvInfo = np.concatenate([
                    EnvInfo_rel_pos_heading,  # 15
                    EnvInfo_speed_limit,  # 1
                    EnvInfo_bounding_box,  # 2
                ], -1).astype(np.float32)

                # Neighbor Info
                on_road_neighbors = []
                for neighbor in current_raw_obs.neighborhood_vehicle_states:
                    if neighbor.lane_id != 'off_lane':
                        on_road_neighbors.append(neighbor)
                on_road_neighbors = on_road_neighbors[:50]
                neighbors_pos = np.zeros([50, 2])
                neighbors_bounding_box = np.zeros([50, 2])
                neighbors_speed = np.zeros(50)
                neighbors_heading = np.zeros([50])
                for nb_ind, neighbor in enumerate(on_road_neighbors):
                    neighbors_pos[nb_ind] = neighbor.position[:2]
                    neighbors_speed[nb_ind] = neighbor.speed
                    neighbors_heading[nb_ind] = neighbor.heading
                    neighbors_bounding_box[nb_ind, 0] = neighbor.bounding_box.length
                    neighbors_bounding_box[nb_ind, 1] = neighbor.bounding_box.width
                nb_num = len(on_road_neighbors)
                neighbors_rel_vel = np.zeros((50, 2))
                neighbors_rel_vel[:nb_num, 0] = -np.sin(neighbors_heading[:nb_num]) * neighbors_speed[:nb_num] + np.sin(
                    heading) * speed
                neighbors_rel_vel[:nb_num, 1] = np.cos(neighbors_heading[:nb_num]) * neighbors_speed[:nb_num] - np.cos(
                    heading) * speed

                nb_mask = np.all(neighbors_pos == 0, -1).astype(np.float32)

                neighbors_dist = np.sqrt(((neighbors_pos - pos) ** 2).sum(-1)) + nb_mask * 1e5
                st = np.argsort(neighbors_dist)[:5]

                NeighborInfo_rel_pos = ((neighbors_pos[st] - pos) @ rotate_M.T)
                NeighborInfo_rel_vel = ((neighbors_rel_vel[st]) @ rotate_M.T)
                NeighborInfo_rel_heading = (neighbors_heading - heading)[st].reshape(5, 1)
                NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading > np.pi)] -= np.pi
                NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading < -np.pi)] += np.pi
                NeighborInfo_boundingbox = neighbors_bounding_box[st]

                NeighborInfo = np.concatenate([
                    NeighborInfo_rel_pos,  # 2
                    NeighborInfo_rel_vel,  # 2
                    NeighborInfo_rel_heading,  # 1
                    NeighborInfo_boundingbox,  # 2
                ], -1).astype(np.float32)

                if len(on_road_neighbors) < 5:  # padding
                    NeighborInfo[len(on_road_neighbors):] = 0

                obs.append(np.concatenate([
                    NeighborInfo.reshape(-1, ),  # (25)
                    EnvInfo.reshape(-1, ),  # (15)
                ]))

                # 确定action：相对速度小于0.2->0，不小于0.2->1
                speed, heading, lane_index, lane_id = state.speed, state.heading, state.lane_index, state.lane_id
                try:
                    speed_limit = current_raw_obs.waypoint_paths[lane_index][0].speed_limit
                    # 有时waymo数据集的lane_index会out of range
                except:
                    speed_limit = current_raw_obs.waypoint_paths[0][0].speed_limit
                if speed / speed_limit >= 0.2:
                    action = 1  # moving
                else:
                    action = 0  # keeping still
                actions.append(action)
                # 确定terminal
                next_raw_obs = vehicle_data[float(sim_time_next)]
                events = next_raw_obs.events
                # 使用next_obs判定是否到达目的地，若到达则terminal
                next_position = vehicle_data[float(sim_time_next)].ego_vehicle_state.position
                goal_reward = goal_region_reward(
                    threshold,
                    goal_pos_x,
                    goal_pos_y,
                    next_position[0],
                    next_position[1],
                )  # 若到达终点则奖励+10，不确定是否要使用这一项
                if float(sim_time) > 8.7:
                    aaa = 1
                is_reached_goal = True if goal_reward > 0 else False
                if (not (events.off_road or events.reached_goal or events.reached_max_episode_steps or
                         is_reached_goal)) and len(events.collisions) == 0:
                    terminal = 0
                else:
                    terminal = 1
                terminals.append(terminal)
                # 确定reward：[distance_travelled, action_reward, goal_reward]
                dist_reward = next_raw_obs.distance_travelled
                action_reward = {
                    0: 0,
                    1: 1
                }[action]
                reward = [dist_reward, action_reward, goal_reward]

                rewards.append(reward)
                episode_terminals.append(0)
                if terminal == 1:
                    break
            episode_terminals[-1] = 1  # 在每个episode的最后一个step，将episode_terminals的最后一个元素修改为1
            print(f'ind {sc_index} scen {scenario} total_len {len(obs) - last_len} ter_sum {sum(episode_terminals)}')
            last_len = len(obs)

    print(str(len(obs)) + " pieces of turning/changing lane data are added into dataset.")
    obs = np.array(obs)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    episode_terminals = np.array(episode_terminals)

    gen_dict = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
        'episode_terminals': [],
    }
    gen_dict['observations'] = obs
    gen_dict['actions'] = actions
    gen_dict['rewards'] = rewards
    gen_dict['terminals'] = terminals
    gen_dict['episode_terminals'] = episode_terminals
    with open('/home/rl/SMARTS/competition/' + 'dataset_1028.pkl', 'wb') as f:
        pickle.dump(gen_dict, f)
        print("Done loading data")
    # exit()
    #
    #
    # with open('/home/rl/SMARTS/competition/' + 'dataset_1028.pkl', 'rb') as f:
    #     gen_dict = pickle.load(f)
    #     print("Done loading data")
    #
    # obs = gen_dict['observations']
    # actions = gen_dict['actions']
    # rewards = gen_dict['rewards']
    # terminals = gen_dict['terminals']
    # episode_terminals = gen_dict['episode_terminals']

    # exit()

    dataset = MDPDataset(obs, actions, rewards, terminals, episode_terminals=episode_terminals)
    # print(dataset.observations[0])
    # print(rewards)
    # exit()


    # print(dataset.actions.shape)
    # exit()

    encoder = lhb_d3rlpy.models.encoders.VectorEncoderFactory([750, 750, 750])
    # RvS需要的goal特征已直接并入obs
    model = lhb_d3rlpy.algos.DiscreteBC(
        use_gpu=False, batch_size=64,encoder_factory=encoder,
    )


    # minimum = [-0.1, 0, -0.1]
    # maximum = [0.1, 2, 0.1]
    # action_scaler = MinMaxActionScaler(minimum=minimum, maximum=maximum)
    # model = lhb_d3rlpy.algos.CQL(
    #     use_gpu=False, batch_size=256, action_scaler=action_scaler, conservative_weight=conservative_weight
    # )

    # 转储模型
    # policy_name = '/home/rl/SMARTS/competition/track2/train/my_train/d3rlpy_logs_debug/CQL_20221010101413/model_100000.pt'
    # # policy_name = './policy_cpu.pt'
    # policy = torch.load(policy_name)
    # model.build_with_dataset(dataset)
    # print(type(model))
    # print(type(policy))
    # model.load_model(policy_name)
    # model.save_policy('./model_100000_newcpu.pt')
    # aaa = 1
    # exit('finish converting')

    model.fit(
        dataset,
        eval_episodes=dataset,
        n_steps_per_epoch=n_steps_per_epoch,
        n_steps=n_steps,
        logdir=save_directory,
        tensorboard_dir='/home/rl/github/SMARTS/competition/track2/train/runs_{}'.format(output_suffix),
        scorers={
            'action_match_score': lhb_d3rlpy.metrics.discrete_action_match_scorer,
        },
        save_interval=10,
    )  # 使程序在训练过程中只保存1个模型
    #
    # saved_models = glob.glob(str(save_directory / "*"))
    # latest_model = max(saved_models, key=os.path.getctime)
    # os.rename(latest_model, str(save_directory / f"{index + 1}"))
    # index += 1
    # shutil.rmtree(save_directory)
    # model.save_policy(os.path.join(output_path, "model.pt"))


def main(args: argparse.Namespace):
    input_path = args.input_dir
    output_suffix = args.output_suffix
    conservative_weight = args.conservative_weight
    train(input_path, output_suffix, conservative_weight)


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--input_dir",
        help="The path to the directory containing the offline training data",
        type=str,
        default="../../../../SMARTS/competition/offline_datasets_1006/",
    )
    parser.add_argument(
        "--output_suffix",
        help="The path to the directory storing the trained model",
        type=str,
        default="debug",
    )
    parser.add_argument(
        "--conservative_weight",
        type=float,
    )

    args = parser.parse_args()

    main(args)
