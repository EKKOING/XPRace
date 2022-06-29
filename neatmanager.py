import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from math import ceil, floor
from time import sleep
from typing import Iterable

import neat
import numpy as np
import pymongo
import wandb
from bson.binary import Binary

wandb.init(project="XPRace", entity="xprace", resume=False)
wandb.config = {
    "pop": 100,
    "bonus_mod": 1.0,
    "time_mod": 1.5,
    "episode_length": 120,
    "completion_mod": 1.3,
    "trial": 7,
    "completion_per_frame_mod": 1.2,
    "track": "shorttrack",
}
config_name = 'config3'

# Output Utils
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


def delete_last_lines(n: int = 1) -> None:
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)


def progress_bar(progress: float, in_progress: float, total: float) -> None:
    left = '['
    right = ']'
    bar_length = 30
    fill = '|'
    in_progress_fill = '>'
    percent = progress / total
    in_percent = in_progress / total
    fill_amt = int(round(percent * bar_length))
    fill_str = ''
    for _ in range(fill_amt):
        fill_str += fill
    in_fill_amt = int(round(in_percent * bar_length))
    for _ in range(in_fill_amt):
        fill_str += in_progress_fill
    for _ in range(bar_length - (fill_amt + in_fill_amt)):
        fill_str += ' '
    print(f'{left}{fill_str}{right} {percent:.2%}')


try:
    with open('creds.json') as f:
        creds = json.load(f)
except FileNotFoundError:
    print('creds.json not found!')
    exit()

db_string = creds['mongodb']
client = pymongo.MongoClient(db_string)
db = client.NEAT


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, config_name)


class EvolveManager:
    generation = 0
    num_workers = 0
    gen_start: datetime
    latest = None

    fit_list = []
    completion_list = []
    bonus_list = []
    time_list = []
    avg_completion_list = []
    current_species_list = []
    runtime_list = [10.0, 10.0]
    avg_speed_list = []
    best_time = 120.0
    prev_best_time = 120.0

    def __init__(self, config_file: str, generation: int = 0):
        self.config_file = config_file
        if generation != 0:
            self.latest = f'NEAT-{generation}'
        self.generation = generation
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_file)
        self.p = neat.Checkpointer.restore_checkpoint(
            self.latest) if self.latest else neat.Population(self.config)
        self.p.config = self.config
        self.p.add_reporter(neat.StdOutReporter(False))
        self.checkpointer = neat.Checkpointer(1, filename_prefix='NEAT-')
        self.p.add_reporter(self.checkpointer)

    def eval_genomes(self, genomes, config):
        self.num_workers = 0
        self.gen_start = datetime.now()
        collection = db.genomes
        individual_num = 0
        self.current_species_list = []
        for genome_id, genome in genomes:
            individual_num += 1
            key = genome.key
            net = neat.nn.RecurrentNetwork.create(genome, config)
            net = Binary(pickle.dumps(net))
            species_id = self.p.species.get_species_id(genome_id)
            if species_id not in self.current_species_list:
                self.current_species_list.append(species_id)
            db_entry = {
                'key': key,
                'genome': net,
                'individual_num': individual_num,
                'generation': self.generation,
                'bonus': 0,
                'completion': 0,
                'time': -1.0,
                'started_eval': False,
                'started_at': None,
                'finished_eval': False,
                'algo': 'NEAT',
                'species': species_id,
                'trial': wandb.config['trial'],
                'avg_speed': 0.0,
                'avg_completion_per_frame': 0.0
            }
            if collection.find_one({'generation': self.generation, 'individual_num': individual_num, 'algo': 'NEAT', 'trial': wandb.config['trial']}):
                collection.update_one(
                    {
                        'generation': self.generation,
                        'individual_num': individual_num,
                        'algo': 'NEAT',
                        'trial': wandb.config['trial'],
                    }, {
                        '$set': {
                            'started_eval': False, 'finished_eval': False, 'bonus': 0, 'completion': 0, 'time': -1.0, 'genome': net, 'started_at': None, 'key': key, 'species': species_id, 'avg_speed': 0.0, 'avg_completion_per_frame': 0.0
                        }
                    }
                )
            else:
                collection.insert_one(db_entry)
        sleep(1)
        first_sleep = True
        secs_passed = 0
        no_alert = True
        secs_since_tg_update = 0
        last_secs_tg = 0
        while True:
            uncompleted_training = collection.count_documents({'generation': self.generation,
                                                               'finished_eval': False, 'algo': 'NEAT', 'trial': wandb.config['trial']})
            started_training = collection.count_documents(
                {'generation': self.generation, 'started_eval': True, 'finished_eval': False, 'algo': 'NEAT', 'trial': wandb.config['trial']})
            finished_training = collection.count_documents(
                {'generation': self.generation, 'finished_eval': True, 'algo': 'NEAT', 'trial': wandb.config['trial']})

            if self.num_workers < started_training:
                self.num_workers = started_training

            self.check_eval_status(collection)

            if uncompleted_training == 0:
                break

            if not first_sleep:
                delete_last_lines(6)
            else:
                first_sleep = False
            print(f'=== {datetime.now().strftime("%H:%M:%S")} ===\n{uncompleted_training} genomes still need to be evaluated\n{started_training} currently being evaluated\n{finished_training} have been evaluated')
            progress_bar(finished_training, started_training, len(genomes))
            secs_tg = (ceil(uncompleted_training * np.mean(self.runtime_list) / (self.num_workers + 1)))
            if secs_tg != last_secs_tg:
                secs_since_tg_update = 0
            last_secs_tg = secs_tg
            secs_tg -= secs_since_tg_update
            secs_since_tg_update += 1
            mins_tg = floor(secs_tg / 60)
            secs_tg = round(secs_tg % 60)
            if secs_tg < 10:
                secs_tg = f'0{secs_tg}'
            secs_passed_str = round(secs_passed % 60)
            if secs_passed_str < 10:
                secs_passed_str = f'0{secs_passed_str}'
            print(
                f'{floor(secs_passed / 60)}:{secs_passed_str} Elapsed - ETA {mins_tg}:{secs_tg} Remaining')
            sleep(1)
            secs_passed += 1
            if self.num_workers == 0 and secs_passed % 60 == 0 and no_alert:
                wandb.alert(
                    title="No Worker Nodes Available",
                    text="No workers currently running.",
                )
                no_alert = False
        fit_list = []
        bonus_list = []
        completion_list = []
        time_list = []
        runtime_list = []
        avg_speed_list = []
        avg_completion_per_frame_list = []
        for genome_id, genome in genomes:
            key = genome.key
            results = collection.find_one(
                {'key': key, 'generation': self.generation, 'trial': wandb.config['trial']})
            fitness = 0.0
            if results == None:
                print(f'No results found for {key}')
                continue
            runtime = results['runtime']
            bonus = results['bonus']
            avg_speed = results['avg_speed']
            avg_speed_list.append(avg_speed)
            avg_speed_bonus = (avg_speed / 1.3) * wandb.config['bonus_mod']
            avg_completion_per_frame = results['avg_completion_per_frame']
            avg_completion_per_frame_list.append(avg_completion_per_frame)
            avg_completion_per_frame_bonus = max((avg_completion_per_frame * 0.5) ** wandb.config['completion_per_frame_mod'], 0.0001)
            avg_completion_per_frame_bonus = 20 - (40 / bonus)
            avg_completion_per_frame_bonus = max(avg_completion_per_frame_bonus, 0)
            bonus_list.append(bonus)
            bonus = bonus * wandb.config['bonus_mod']
            bonus = max(bonus, 0.0001)
            bonus = 20 - (40 / bonus)
            bonus = max(bonus, 0)
            completion = max(results['completion'], 0.1)
            completion_list.append(completion)
            completion_bonus =  completion ** wandb.config['completion_mod']
            time = results['time']
            time_bonus = 0.0
            if time > 0:
                time_bonus = max((121.0 - time), 1.0) ** wandb.config['time_mod']
            fitness = time_bonus + completion_bonus + avg_completion_per_frame_bonus
            try:
                genome.fitness = fitness.real
            except AttributeError:
                genome.fitness = float(fitness)
            fit_list.append(fitness)
            runtime_list.append(runtime)
            if time > 0:
                time_list.append(time)
        
        self.prev_best_time = self.best_time
        if len(time_list) > 0 and np.max(time_list) < self.best_time:
            self.best_time = np.max(time_list)
            print(f'New best time: {self.best_time}')

        self.completion_list = completion_list
        self.fit_list = fit_list
        self.bonus_list = bonus_list
        self.time_list = time_list
        self.runtime_list = runtime_list
        self.avg_speed_list = avg_speed_list
        self.avg_completion_list = avg_completion_per_frame_list

    def run(self, num_gens: int = 0):
        
        winner = self.p.run(self.eval_genomes, num_gens)
        return winner

    def check_eval_status(self, collection):
        for genome in collection.find({'generation': self.generation, 'started_eval': True, 'finished_eval': False, 'trial': wandb.config['trial']}):
            genome_id = genome['_id']
            key = genome['key']
            started_at = genome['started_at']
            if (datetime.now() - started_at) > timedelta(minutes=3):
                try:
                    if genome['failed_eval']:
                        print(f'Genome {key} failed to evaluate')
                        collection.update_one({'_id': genome_id}, {'$set': {'finished_eval': True}})
                        wandb.alert(title='Failed Eval!', text=f'Gen: {self.generation} Genome {key} failed to evaluate!')
                        continue
                except KeyError:
                    pass
                delete_last_lines(5)
                print(
                    f'Genome {key} has been running for 3 minutes, marking for review!\n\n\n\n\n\n\n')
                collection.update_one({'_id': genome_id}, {
                                      '$set': {'started_eval': False, 'failed_eval': True}})
                wandb.alert(
                    title="Evaluation Timeout",
                    text=f"Gen: {self.generation} Genome {key} has been running for 3 minutes, marking for review",
                )

manager = EvolveManager(config_path, generation=0)
while True:
    try:
        manager.run(num_gens=1)
        log = {
            "Generation": manager.generation,
            "Avg Fitness": np.mean(manager.fit_list),
            "Median Fitness": np.median(manager.fit_list),
            "Max Fitness": np.max(manager.fit_list),
            "Min Fitness": np.min(manager.fit_list),
            "SD Fitness": np.std(manager.fit_list),
            "Avg Bonus": np.mean(manager.bonus_list),
            "Median Bonus": np.median(manager.bonus_list),
            "Max Bonus": np.max(manager.bonus_list),
            "Min Bonus": np.min(manager.bonus_list),
            "SD Bonus": np.std(manager.bonus_list),
            "Avg Completion": np.mean(manager.completion_list),
            "Median Completion": np.median(manager.completion_list),
            "Max Completion": np.max(manager.completion_list),
            "Min Completion": np.min(manager.completion_list),
            "SD Completion": np.std(manager.completion_list),
            "Avg Runtime": np.mean(manager.runtime_list),
            "Median Runtime": np.median(manager.runtime_list),
            "Max Runtime": np.max(manager.runtime_list),
            "Min Runtime": np.min(manager.runtime_list),
            "SD Runtime": np.std(manager.runtime_list),
            "Avg Speed": np.mean(manager.avg_speed_list),
            "Median Speed": np.median(manager.avg_speed_list),
            "Max Speed": np.max(manager.avg_speed_list),
            "Min Speed": np.min(manager.avg_speed_list),
            "SD Speed": np.std(manager.avg_speed_list),
            "Avg Completion Per Frame": np.mean(manager.avg_completion_list),
            "Median Completion Per Frame": np.median(manager.avg_completion_list),
            "Max Completion Per Frame": np.max(manager.avg_completion_list),
            "Min Completion Per Frame": np.min(manager.avg_completion_list),
            "SD Completion Per Frame": np.std(manager.avg_completion_list),
            "Time Elapsed": (datetime.now() - manager.gen_start).total_seconds(),
            "Time": datetime.now(),
            "Num Species": len(manager.current_species_list),
            "Population Size": len(manager.fit_list),
            "Num Workers": manager.num_workers,
        }
        if len(manager.time_list) != 0:
            log["Avg Time"] = np.mean(manager.time_list)
            log["Median Time"] = np.median(manager.time_list)
            log["Max Time"] = np.max(manager.time_list)
            log["Min Time"] = np.min(manager.time_list)
            log["SD Time"] = np.std(manager.time_list)
        
        if manager.best_time != manager.prev_best_time:
            log["Best Time"] = manager.best_time
        wandb.log(log)
        manager.generation += 1
    except KeyboardInterrupt:
        # Allows time for non-zero exit code with ctrl+c
        sleep(5)

        wandb.alert(
            title="Run Aborted",
            text=f"Run Ended at generation {manager.generation}",
        )
        break
    except Exception as e:
        wandb.alert(
            title="Run Error",
            text=f"Run error at generation {manager.generation} with error {e}",
        )
        print(e)
        raise e
