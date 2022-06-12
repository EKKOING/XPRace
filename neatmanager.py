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

wandb.init(project="XPRace", entity="xprace", resume=True)
wandb.config = {
    "pop": 100,
    "bonus_mod": 0.98,
    "time_mod": 1.5,
    "episode_length": 60,
    "completion_mod": 1.5,
    "trial": 1,
}

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
config_path = os.path.join(local_dir, 'config')

# TODO: load generation number from mongodb


class EvolveManager:
    generation = 0
    num_workers = 0
    gen_start: datetime

    fit_list = []
    completion_list = []
    bonus_list = []
    time_list = []
    current_species_list = []
    runtime_list = []

    def __init__(self, config_file: str, generation: int = 0):
        self.config_file = config_file
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
                'species': species_id
            }
            if collection.find_one({'generation': self.generation, 'individual_num': individual_num, 'algo': 'NEAT'}):
                collection.update_one(
                    {
                        'generation': self.generation,
                        'individual_num': individual_num
                    }, {
                        '$set': {
                            'started_eval': False, 'finished_eval': False, 'bonus': 0, 'completion': 0, 'time': -1.0, 'genome': net, 'started_at': None, 'key': key, 'species': species_id
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
                                                               'finished_eval': False, 'algo': 'NEAT'})
            started_training = collection.count_documents(
                {'generation': self.generation, 'started_eval': True, 'finished_eval': False, 'algo': 'NEAT'})
            finished_training = collection.count_documents(
                {'generation': self.generation, 'finished_eval': True, 'algo': 'NEAT'})

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
            secs_tg = (ceil(uncompleted_training *
                       wandb.config['episode_length'] / (self.num_workers + 1)))
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
        for genome_id, genome in genomes:
            key = genome.key
            results = collection.find_one(
                {'key': key, 'generation': self.generation})
            ## TODO: add bonus and completion to results
            fitness = 0.0
            bonus = results['bonus'] * wandb.config['bonus_mod']
            completion = results['completion'] ** wandb.config['completion_mod']
            time = results['time']
            runtime = results['run_time']
            time_bonus = 0.0
            if time > 0:
                time_bonus = (120.0 - time) ** wandb.config['time_mod']
            fitness = bonus + time_bonus + completion
            genome.fitness = fitness
            fit_list.append(fitness)
            bonus_list.append(bonus)
            completion_list.append(completion)
            runtime_list.append(runtime)
            if time > 0:
                time_list.append(time)

        self.completion_list = completion_list
        self.fit_list = fit_list
        self.bonus_list = bonus_list
        self.time_list = time_list
        self.runtime_list = runtime_list

    def run(self, num_gens: int = 0):
        
        winner = self.p.run(self.eval_genomes, num_gens)
        return winner

    def check_eval_status(self, collection):
        for genome in collection.find({'generation': self.generation, 'started_eval': True, 'finished_eval': False}):
            genome_id = genome['_id']
            key = genome['key']
            started_at = genome['started_at']
            if (datetime.now() - started_at) > timedelta(minutes=2):
                delete_last_lines(5)
                print(
                    f'Genome {key} has been running for 2 minutes, marking for review!\n\n\n\n\n\n\n')
                collection.update_one({'_id': genome_id}, {
                                      '$set': {'started_eval': False}})
                wandb.alert(
                    title="Evaluation Timeout",
                    text=f"Genome {key} has been running for 5 minutes, marking for review",
                )

manager = EvolveManager(config_path, generation=381)
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
            "Time Elapsed": (datetime.now() - manager.gen_start).total_seconds(),
            "Time": datetime.now(),
            "Num Species": len(manager.current_species_list),
        }
        if len(manager.time_list) != 0:
            log["Avg Time"] = np.mean(manager.time_list)
            log["Median Time"] = np.median(manager.time_list)
            log["Max Time"] = np.max(manager.time_list)
            log["Min Time"] = np.min(manager.time_list)
            log["SD Time"] = np.std(manager.time_list)
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
