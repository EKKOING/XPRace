import numpy as np
default_config = {
    "bonus_mod": 1.0,
    "time_mod": 1.2,
    "episode_length": 120,
    "completion_mod": 1.1,
    "completion_per_frame_mod": 1.2,
}


def get_fitness(completion: float, bonus: float, time: float, avg_speed: float, avg_completion_per_frame: float, target_time: float, max_target_time: float = 0.0, config: dict = default_config) -> float:
    completion_bonus =  completion ** config['completion_mod']
    time_bonus = 0.0
    time_adj = 1.0
    if max_target_time != 0.0:
        time_adj = max_target_time / target_time
    if time > 0:
        time_bonus = ((target_time * time_adj) - (min(time, target_time - 1.0) * time_adj)) ** config['time_mod'] + 100 ** config['completion_mod']
    bonus = min((bonus * time_adj), 50.0)
    fitness = time_bonus + completion_bonus + bonus
    try:
        fitness = fitness.real
    except AttributeError:
        fitness = float(fitness)
    return fitness

def get_many_fitnesses(completions: list, bonuses: list, times: list, avg_speeds: list, avg_completions_per_frame: list, target_times: list, config: dict = default_config) -> list:
    fitnesses = []
    for i in range(len(completions)):
        fitnesses.append(get_fitness(completions[i], bonuses[i], times[i], avg_speeds[i], avg_completions_per_frame[i], target_times[i], np.max(target_times), config))
    return fitnesses

if __name__ == "__main__":
    print(get_fitness(0.0, 0.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(50.0, 14.0, 0.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(50.0, 20.0, 0.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(70.0, 20.0, 0.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(100.0, 20.0, 55.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(100.0, 20.0, 50.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(100.0, 20.0, 45.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(100.0, 25.0, 45.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(100.0, 50.0, 1.0, 0.0, 0.0, 60.0, 60.0))
    print(get_fitness(0.0, 0.0, 0.0, 0.0, 30.0, 60.0))
    print(get_fitness(50.0, 14.0, 0.0, 0.0, 0.0, 30.0, 60.0))
    print(get_fitness(50.0, 20.0, 0.0, 0.0, 0.0, 30.0, 60.0))
    print(get_fitness(70.0, 20.0, 0.0, 0.0, 0.0, 30.0, 60.0))
    print(get_fitness(100.0, 20.0, 27.5, 0.0, 0.0, 30.0, 60.0))
    print(get_fitness(100.0, 20.0, 25.0, 0.0, 0.0, 30.0, 60.0))
    print(get_fitness(100.0, 20.0, 22.5, 0.0, 0.0, 30.0, 60.0))
    print(get_fitness(100.0, 25.0, 22.5, 0.0, 0.0, 30.0, 60.0))
    print(get_fitness(100.0, 50.0, 1.0, 0.0, 0.0, 30.0, 60.0))