"""Run a trained race-car driver in the interactive simulator."""

import argparse
from functools import partial
from math import isfinite
from pathlib import Path
from time import sleep

from gymnasium import ActionWrapper
from stable_baselines3 import DQN, PPO

from Agents.SB3Drivers import (
    DrivableTrackDriver,
    LocalDrivableTrackDriver,
    LocalTrackDQNDriver,
    LocalTrackPPODriver,
    SB3Driver,
    SB3ObservationWrapper,
    TrackDQNDriver,
    TrackPPODriver,
)
from Agents.wrappers import (
    DrivableTrackObservationWrapper,
    FlattenActionWrapper,
    LocalDrivableTrackObservationWrapper,
    LocalTrackObservationWrapper,
    SpeedLimitActionWrapper,
    TrackObservationWrapper,
)


PROJECT_DIR = Path(__file__).resolve().parent
TRACK_DRIVERS = {
    'track-ppo': (TrackPPODriver, 'track_ppo_allow_collisions.zip'),
    'track-dqn': (TrackDQNDriver, 'track_dqn_allow_collisions.zip'),
    'track-ppo-no-colli': (TrackPPODriver, 'track_ppo_no_collisions.zip'),
    'local-track-ppo': (LocalTrackPPODriver, 'local_track_ppo_allow_collisions.zip'),
    'local-track-dqn': (LocalTrackDQNDriver, 'local_track_dqn_allow_collisions.zip'),
    'drivable-track-ppo': (partial(DrivableTrackDriver, model_class=PPO), 'drivable_track_ppo_no_collisions.zip'),
    'drivable-track-dqn': (partial(DrivableTrackDriver, model_class=DQN), 'drivable_track_dqn_no_collisions.zip'),
    'local-drivable-track-ppo': (
        partial(LocalDrivableTrackDriver, model_class=PPO),
        'local_drivable_track_ppo_no_collisions.zip',
    ),
    'local-drivable-track-dqn': (
        partial(LocalDrivableTrackDriver, model_class=DQN),
        'local_drivable_track_dqn_no_collisions.zip',
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--driver', choices=TRACK_DRIVERS,
        help='track uses boundaries, drivable-track uses a drivable-area mask, and local views follow the car.',
    )
    parser.add_argument('--model', type=Path, help='load a checkpoint from this path')
    parser.add_argument(
        '--max-speed', type=float, metavar='M/S',
        help='cut throttle above this positive speed (default: no limit)',
    )
    parser.add_argument(
        '--full-track-frame', choices=('track', 'world'),
        help='use the cropped track frame or the original world frame used by older checkpoints',
    )
    parser.add_argument(
        '--legacy-local-distance', action='store_true',
        help='restore the fixed distance channel for an older local-track checkpoint',
    )
    parser.add_argument('--show-track-view', action='store_true', help='play the track channels seen by the model')
    parser.add_argument(
        '--compare-track-views', action='store_true',
        help='show the model input and the other track view from the same car pose',
    )
    args = parser.parse_args()
    if args.max_speed is not None and (not isfinite(args.max_speed) or args.max_speed <= 0):
        parser.error('--max-speed must be a finite positive number in m/s')
    show_track_view = args.show_track_view or args.compare_track_views
    if show_track_view and args.driver is None:
        parser.error('track view playback requires --driver with a track or local-track model')
    driver_name = args.driver or ''
    is_dqn = driver_name.endswith('-dqn')
    is_local = driver_name.startswith('local-')
    is_drivable = driver_name.startswith(('drivable-track-', 'local-drivable-track-'))
    is_full_track = bool(driver_name) and not is_local
    if args.full_track_frame is not None and not is_full_track:
        parser.error('--full-track-frame requires a full-track driver')
    if args.legacy_local_distance and not is_local:
        parser.error('--legacy-local-distance requires a local-track driver')
    full_track_frame = args.full_track_frame
    if full_track_frame is None:
        full_track_frame = 'track' if args.model is not None or is_drivable else 'world'

    if args.driver is None:
        driver_class = partial(SB3Driver, model_class=PPO, wrappers=[FlattenActionWrapper, SB3ObservationWrapper])
        model = 'ppo_model_w_collision.zip'
        scenario = 'austria.yml'
    else:
        driver_class, model = TRACK_DRIVERS[args.driver]
        scenario = 'austria_allow_collisions.yml'

    model_path = args.model if args.model is not None else PROJECT_DIR / 'models' / model
    if args.model is None and args.driver is not None and not model_path.is_file():
        legacy_path = model_path.with_name(f'{model_path.stem}_old{model_path.suffix}')
        if legacy_path.is_file():
            model_path = legacy_path
    if not model_path.is_file():
        parser.error(f'model file not found: {model_path}')

    from racecar_gym.envs.gym_api import SingleAgentRaceEnv

    inference_kwargs = {'device': 'cpu'}
    if is_dqn:
        # DQN creates a replay buffer when loading, even for inference.
        inference_kwargs['buffer_size'] = 1
    driver_kwargs = {'crop_to_track': full_track_frame == 'track'} if is_full_track else {}
    if is_local:
        driver_kwargs['include_distance'] = args.legacy_local_distance or (args.model is None and not is_drivable)

    env = SingleAgentRaceEnv(scenario=str(PROJECT_DIR / 'scenarios' / scenario), render_mode='human')
    display = None
    try:
        if args.max_speed is not None:
            env = SpeedLimitActionWrapper(env, max_speed=args.max_speed)
        driver = driver_class(env, _init_setup_model=False, **inference_kwargs, **driver_kwargs)
        env = driver.env
        driver.model = driver.model_class.load(model_path, env=env, **inference_kwargs)

        track_key = DrivableTrackObservationWrapper.KEY if is_drivable else TrackObservationWrapper.KEY
        comparison = None
        if show_track_view:
            from track_view_display import TrackViewDisplay

            if is_local:
                model_view = 'local track'
            else:
                model_view = 'cropped full track' if full_track_frame == 'track' else 'original full track'
            view_names = [f'Model input ({model_view})']
            channel_counts = [env.observation_space[track_key].shape[0]]
            if args.compare_track_views:
                if is_local:
                    comparison_class = DrivableTrackObservationWrapper if is_drivable else TrackObservationWrapper
                    other_view = 'cropped full track'
                else:
                    comparison_class = LocalDrivableTrackObservationWrapper if is_drivable else LocalTrackObservationWrapper
                    other_view = 'local track'
                comparison = comparison_class(env.unwrapped)
                view_names.append(f'Comparison ({other_view})')
                channel_counts.append(comparison.observation_space[track_key].shape[0])

            action_wrapper = env
            while not isinstance(action_wrapper, ActionWrapper):
                action_wrapper = action_wrapper.env
            mask_label = 'Drivable area' if is_drivable else 'Track boundary'
            display = TrackViewDisplay(view_names, channel_counts, mask_label=mask_label)

        obs, _ = env.reset(options={'mode': 'grid'})
        done = truncated = False
        step = 0
        while not (done or truncated):
            action, _ = driver.model.predict(obs, deterministic=args.driver is not None)
            if display is not None and display.is_open:
                views = [obs[track_key]]
                if comparison is not None:
                    views.append(comparison.observation(obs)[track_key])
                steering = action_wrapper.action(action)['steering'].item()
                display.update(views, step, steering)
            obs, _, done, truncated, _ = env.step(action)
            step += 1
            sleep(0.01)
    finally:
        if display is not None:
            display.close()
        env.close()


if __name__ == '__main__':
    main()
