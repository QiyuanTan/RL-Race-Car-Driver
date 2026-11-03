"""Run FTG and save the results to JSON. Use --render to watch the car."""
import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import racecar_gym.envs.gym_api  # noqa: F401  (register the environments)

from Agents.FollowTheGapDriver import FollowTheGapDriver

# Shared environment name for the runner and saved results.
ENV_ID = 'SingleAgentAustria-v0'


def run_episode(env, driver, seed):
    """Run one episode and return its lap time, contacts, and progress."""
    # Seed numpy too, since the LiDAR noise uses its global random generator.
    np.random.seed(seed)
    obs, info = env.reset(seed=seed, options=dict(mode='grid'))
    driver.reset()

    lap_time = None  # stays None if the car never finishes a lap
    collisions = 0
    touching_wall = False
    progress = 0.0  # net progress from the starting position, measured in laps
    last_progress = info['progress']
    done = truncated = False
    previous_motor = None
    previous_steering = None
    motor_change_sum = 0.0
    steering_change_sum = 0.0
    change_count = 0

    while not (done or truncated):

        action = driver.act(obs) # action of the driver
        motor = float(action['motor'][0]) # current motor command
        steering = float(action['steering'][0]) # current steering command

        # check with the previous commands/action
        if previous_motor is not None:
            motor_change_sum += abs(motor - previous_motor)
            steering_change_sum += abs(steering - previous_steering)
            change_count += 1
        # save the action for next
        previous_motor = motor
        previous_steering = steering

        obs, reward, done, truncated, info = env.step(action) # execute the action in the simulator

        # Count a new contact once. Staying against the wall does not add more.
        if info['wall_collision'] and not touching_wall:
            collisions += 1
        touching_wall = info['wall_collision']

        # Crossing the start line wraps progress from ~1 to ~0.
        # Undo that jump before adding the change; going backwards subtracts progress.
        change = info['progress'] - last_progress
        if change < -0.5:
            change += 1.0
        elif change > 0.5:
            change -= 1.0
        progress += change
        last_progress = info['progress']

        # Use the environment's lap count, which checks checkpoint order.
        # Record the first finish time, then let the environment end the episode.
        if info['lap'] >= 2 and lap_time is None:
            lap_time = round(float(info['time']), 2)

    # calculate average changes between consecutive actions
    if change_count > 0:
        motor_smoothness = motor_change_sum / change_count
        steering_smoothness = steering_change_sum / change_count
    else:
        motor_smoothness = None
        steering_smoothness = None

    # round the smoothness values, only if they exist
    if motor_smoothness is not None:
        motor_smoothness = round(motor_smoothness, 6)
    if steering_smoothness is not None:
        steering_smoothness = round(steering_smoothness, 6)

    # check if the lap was finished
    if lap_time is not None:
        lap_completed = True
    else:
        lap_completed = False

    # round progress and simulation time
    progress = round(float(progress), 3)
    sim_time = round(float(info['time']), 2)

    # build the results dictionary
    results = {
        "seed": seed,
        "lap_completed": lap_completed,
        "lap_time": lap_time,
        "collisions": collisions,
        "progress": progress,
        "sim_time": sim_time,
        "motor_smoothness": motor_smoothness,
        "steering_smoothness": steering_smoothness,
    }
    return results


def run_episode_in_new_process(job):
    """create the environment and driver for one job, then close after running."""
    params, seed, render = job
    env = gym.make(ENV_ID, render_mode='human' if render else 'rgb_array_birds_eye')
    result = run_episode(env, FollowTheGapDriver(env, **params), seed)
    env.close()
    return result


def parse_params(pairs):
    """turn --param entries like max_speed=3.5 into driver settings."""
    params = {}
    for pair in pairs or []:
        name, value = pair.split('=')
        params[name] = float(value)
    return params


def summarize(episodes):
    """combine the episode results into one summary."""
    # Only finished laps have a lap time. Collision stats include every episode.
    laps = [e['lap_time'] for e in episodes if e['lap_completed']]


    # get all motor smoothness values that exist
    motor_values = [] # start with empty list
    for i in episodes: # go through each episode one by one
        value = i['motor_smoothness'] # get this episode's motor smoothness
        if value is not None: # only keep it if it actually exists
            motor_values.append(value) # add it to the list

    # get all steering smoothness values that exist
    steering_values = []
    for j in episodes:
        value = j['steering_smoothness']
        if value is not None:
            steering_values.append(value)


    # calculate the average motor smoothness
    if len(motor_values) > 0:
        motor_mean = np.mean(motor_values) # take the average
        motor_mean = float(motor_mean) # turn it into a normal Python number
        motor_smoothness_mean = round(motor_mean, 6) # round to 6 decimal places
    else:
        motor_smoothness_mean = None

    # Calculate the average steering smoothness
    if len(steering_values) > 0:
        steering_mean = np.mean(steering_values)
        steering_mean = float(steering_mean)
        steering_smoothness_mean = round(steering_mean, 6) # round to 6 decimal
    else:
        steering_smoothness_mean = None

    return dict(
        env=ENV_ID,
        seeds=[e['seed'] for e in episodes],
        laps_completed=f"{len(laps)}/{len(episodes)}",
        collision_free_laps=sum(1 for e in episodes if e['lap_completed'] and e['collisions'] == 0),
        lap_time_mean=round(float(np.mean(laps)), 2) if laps else None,
        lap_time_std=round(float(np.std(laps)), 2) if laps else None,
        lap_time_best=round(min(laps), 2) if laps else None,
        lap_time_worst=round(max(laps), 2) if laps else None,
        collisions_mean=float(np.mean([e['collisions'] for e in episodes])),
        motor_smoothness_mean=motor_smoothness_mean,        # use the value calculated above
        steering_smoothness_mean=steering_smoothness_mean,  # use the value calculated above
    )


def main():
    """Read the run options, launch episodes, and save their results."""
    # These options control the experiment. --param changes the driver's settings.
    parser = argparse.ArgumentParser(description='Evaluate the Follow-the-Gap baseline.')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100, help='episode i uses seed + i')
    parser.add_argument('--workers', type=int, default=1, help='episodes run in parallel')
    parser.add_argument('--param', action='append', help='change a driver parameter, e.g. max_speed=3.5')
    parser.add_argument('--render', action='store_true', help='show the PyBullet window (1 worker)')
    parser.add_argument('--out', default='results/baseline_ftg')
    args = parser.parse_args()

    params = parse_params(args.param)
    jobs = [(params, args.seed + i, args.render) for i in range(args.episodes)]
    workers = 1 if args.render else args.workers  # one window at a time

    episodes = []
    #  use a fresh process per episode to avoid leftover simulator state.
    # spawn starts a clean process; maxtasksperchild=1 replaces it after one job.
    with mp.get_context('spawn').Pool(workers, maxtasksperchild=1) as pool:
        for result in pool.imap(run_episode_in_new_process, jobs):
            print(result, flush=True)
            episodes.append(result)

    summary = summarize(episodes)
    print(json.dumps(summary, indent=2))

    # Keep the settings, summary, and individual runs together for later comparison.
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"ftg_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, 'w') as f:
        json.dump(dict(params=params or 'defaults', summary=summary, episodes=episodes), f, indent=2)
    print('saved', path)


# Keep imports from starting experiments, including in spawned worker processes.
if __name__ == '__main__':
    main()