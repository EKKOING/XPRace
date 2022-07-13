default_config = {
    "bonus_mod": 1.0,
    "time_mod": 1.5,
    "episode_length": 120,
    "completion_mod": 1.3,
    "completion_per_frame_mod": 1.2,
}


def get_fitness(completion: float, bonus: float, time: float, avg_speed: float, avg_completion_per_frame: float, config: dict = default_config) -> float:
    avg_speed_bonus = (avg_speed / 1.3) * config['bonus_mod']
    avg_completion_per_frame_bonus = max(avg_completion_per_frame, 0.0001)
    avg_completion_per_frame_bonus = 15.0 - (2.0 / avg_completion_per_frame_bonus)
    avg_completion_per_frame_bonus = max(avg_completion_per_frame_bonus, 0)
    bonus = bonus * config['bonus_mod']
    bonus = max(bonus, 0.0001)
    bonus = 30.0 - (20.0 / bonus)
    bonus = max(bonus, 0)
    completion_bonus =  completion ** config['completion_mod']
    time_bonus = 0.0
    if time > 0:
        time_bonus = max((121.0 - time), 1.0) ** config['time_mod']
    fitness = time_bonus + completion_bonus + avg_completion_per_frame_bonus + bonus
    return fitness