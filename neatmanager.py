import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timedelta
from math import ceil, floor
from time import sleep

import neat
import numpy as np
import pymongo
from bson.binary import Binary

import wandb
from consoleutils import delete_last_lines, progress_bar
from xpracefitness import get_fitness, get_many_fitnesses

wandb.init(project="XPRace", entity="xprace", resume="must", id="c5i9zwx6")
##wandb.init(project="XPRace", entity="xprace", resume=False)
wandb.config = {
    "pop": 200,
    "bonus_mod": 1.0,
    "time_mod": 1.2,
    "episode_length": 120,
    "completion_mod": 1.1,
    "trial": 25,
    "completion_per_frame_mod": 1.2,
    "tracks": ["circuit1_a", "circuit1_b"],
}
config_name = "config4"

input(f'Confirm Trial: {wandb.config["trial"]} and {config_name}')

try:
    with open("creds.json") as f:
        creds = json.load(f)
except FileNotFoundError:
    print("creds.json not found!")
    exit()

target_times = []
for track in wandb.config["tracks"]:
    try:
        with open(f"{track}.json") as f:
            track = json.load(f)
            target_times.append(track["target_time"])
    except FileNotFoundError:
        print(f"{track}.json not found!")
        exit()

db_string = creds["mongodb"]
client = pymongo.MongoClient(db_string)
db = client.NEAT


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, config_name)


class EvolveManager:
    generation = 0
    num_workers = 0
    gen_start: datetime
    latest = None

    summed_fit_list = [0.0]
    fit_list = np.empty((0, 2))
    bonus_list = np.empty((0, 2))
    completion_list = np.empty((0, 2))
    time_list = np.empty((0, 2))
    runtime_list = np.ones((2, 2)) * 10.0
    avg_speed_list = np.empty((0, 2))
    avg_completion_per_frame_list = np.empty((0, 2))
    frame_list = np.empty((0, 2))
    end_frame_list = np.empty((0, 2))
    autopsy_list = [{} for _ in range(2)]
    fitness_weight = [0.5]

    failed_evals = 0
    timedout_evals = 0
    low_framerate = 0

    def __init__(self, config_file: str, generation: int = 0):
        self.config_file = config_file
        if generation != 0:
            self.latest = f"NEAT-{generation}"
        self.generation = generation
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )
        self.p = (
            neat.Checkpointer.restore_checkpoint(self.latest)
            if self.latest
            else neat.Population(self.config)
        )
        self.p.config = self.config
        self.p.add_reporter(neat.StdOutReporter(False))
        self.checkpointer = neat.Checkpointer(1, filename_prefix="NEAT-")
        self.p.add_reporter(self.checkpointer)
        self.num_tracks = len(wandb.config["tracks"])

    def eval_genomes(self, genomes, config):
        print(
            f'=== {datetime.now().strftime("%H:%M:%S")} === Starting Eval Cycle > {self.generation}'
        )
        self.num_workers = 0
        self.gen_start = datetime.now()
        collection = db.genomes
        individual_num = 0
        self.current_species_list = []
        self.timedout_evals = 0
        self.failed_evals = 0
        self.low_framerates = 0
        for genome_id, genome in genomes:
            individual_num += 1
            key = genome.key
            net = neat.nn.RecurrentNetwork.create(genome, config)
            net = Binary(pickle.dumps(net))
            species_id = self.p.species.get_species_id(genome_id)
            if species_id not in self.current_species_list:
                self.current_species_list.append(species_id)
            db_entry = {
                "key": key,
                "genome": net,
                "individual_num": individual_num,
                "generation": self.generation,
                "tracks": wandb.config["tracks"],
                "bonus": np.zeros(self.num_tracks).tolist(),
                "completion": np.zeros(self.num_tracks).tolist(),
                "time": (np.ones(self.num_tracks) * -1).tolist(),
                "started_eval": False,
                "started_at": None,
                "finished_eval": False,
                "algo": "NEAT",
                "species": species_id,
                "trial": wandb.config["trial"],
                "avg_speed": np.zeros(self.num_tracks).tolist(),
                "avg_completion_per_frame": np.zeros(self.num_tracks).tolist(),
                "failed_eval": False,
            }
            collection.find_one_and_replace(
                {
                    "generation": self.generation,
                    "individual_num": individual_num,
                    "algo": "NEAT",
                    "trial": wandb.config["trial"],
                },
                replacement=db_entry,
                upsert=True,
            )
        sleep(1)
        first_sleep = True
        secs_passed = 0
        no_alert = True
        secs_since_tg_update = 0
        last_secs_tg = 0
        avg_eval_time = np.mean(np.sum(self.runtime_list, axis=1)) + (
            8.0 * self.num_tracks
        )
        start_time = datetime.now()
        while True:
            uncompleted_training = collection.count_documents(
                {
                    "generation": self.generation,
                    "finished_eval": False,
                    "algo": "NEAT",
                    "trial": wandb.config["trial"],
                }
            )
            started_training = collection.count_documents(
                {
                    "generation": self.generation,
                    "started_eval": True,
                    "finished_eval": False,
                    "algo": "NEAT",
                    "trial": wandb.config["trial"],
                }
            )
            finished_training = collection.count_documents(
                {
                    "generation": self.generation,
                    "finished_eval": True,
                    "algo": "NEAT",
                    "trial": wandb.config["trial"],
                }
            )

            self.num_workers = len(collection.distinct('hostname', {'trial': wandb.config["trial"], 'generation': self.generation, "algo": "NEAT"}))


            self.check_eval_status(collection)

            if uncompleted_training == 0:
                break

            if not first_sleep:
                delete_last_lines(6)
            else:
                first_sleep = False
            print(
                f'=== {datetime.now().strftime("%H:%M:%S")} ===\n{uncompleted_training} genomes still need to be evaluated\n{started_training} currently being evaluated\n{finished_training} have been evaluated'
            )
            progress_bar(finished_training, started_training, len(genomes))
            secs_tg = ceil(
                uncompleted_training * (avg_eval_time) / (self.num_workers + 1)
            )
            if uncompleted_training - started_training > 0:
                secs_tg = max(avg_eval_time, secs_tg)
            if secs_tg != last_secs_tg:
                secs_since_tg_update = 0
            last_secs_tg = secs_tg
            secs_tg -= secs_since_tg_update
            secs_since_tg_update += 1
            mins_tg = floor(secs_tg / 60)
            secs_tg = round(secs_tg % 60)
            if secs_tg < 10:
                secs_tg = f"0{secs_tg}"
            secs_passed_str = round(secs_passed % 60)
            if secs_passed_str < 10:
                secs_passed_str = f"0{secs_passed_str}"
            print(
                f"{floor(secs_passed / 60)}:{secs_passed_str} Elapsed - ETA {mins_tg}:{secs_tg} Remaining"
            )
            sleep(1)
            secs_passed = (datetime.now() - start_time).total_seconds()
            if self.num_workers == 0 and secs_passed % 60 == 0 and no_alert:
                wandb.alert(
                    title="No Worker Nodes Available",
                    text="No workers currently running.",
                )
                no_alert = False

        print(
            f'=== {datetime.now().strftime("%H:%M:%S")} === Finished Evaluating Genomes > {self.generation}'
        )

        summed_fit_list = []
        fit_list = np.empty((0, self.num_tracks))
        bonus_list = np.empty((0, self.num_tracks))
        completion_list = np.empty((0, self.num_tracks))
        time_list = np.empty((0, self.num_tracks))
        runtime_list = np.empty((0, self.num_tracks))
        avg_speed_list = np.empty((0, self.num_tracks))
        avg_completion_per_frame_list = np.empty((0, self.num_tracks))
        frame_list = np.empty((0, self.num_tracks))
        end_frame_list = np.empty((0, self.num_tracks))
        possible_autopsies = {
            "Collision": 0,
            "Stuck": 0,
            "Time": 0,
            "Completed": 0,
            "Failed to start": 0,
        }
        autopsy_list = [possible_autopsies for _ in range(self.num_tracks)]
        fitness_weight = []
        for genome_id, genome in genomes:
            key = genome.key
            results = collection.find_one(
                {
                    "trial": wandb.config["trial"],
                    "generation": self.generation,
                    "key": key,
                }
            )
            if results == None:
                print(f"No results found for {key}")
                continue
            runtime = results["runtime"]
            completion = results["completion"]
            bonus = results["bonus"]
            avg_speed = results["avg_speed"]
            avg_completion_per_frame = results["avg_completion_per_frame"]
            time = results["time"]
            frame = results["frame"]
            autopsies = results["autopsy"]
            end_frame = results["end_frame"]
            for i, autopsy in enumerate(autopsies):
                if autopsy in autopsy_list[i]:
                    autopsy_list[i][autopsy] += 1
                else:
                    autopsy_list[i][autopsy] = 1
            avg_speed_list = np.append(avg_speed_list, [avg_speed], axis=0)
            avg_completion_per_frame_list = np.append(
                avg_completion_per_frame_list, [avg_completion_per_frame], axis=0
            )
            completion_list = np.append(completion_list, [completion], axis=0)
            bonus_list = np.append(bonus_list, [bonus], axis=0)

            fitnesses = get_many_fitnesses(
                completion,
                bonus,
                (np.array(frame, dtype=float) / 28.0).tolist(),
                avg_speed,
                avg_completion_per_frame,
                target_times,
                wandb.config,
            )
            genome.fitness = np.sum(fitnesses)
            collection.update_one(
                {
                    "key": key,
                    "generation": self.generation,
                    "trial": wandb.config["trial"],
                },
                {"$set": {"fitness": genome.fitness}},
            )
            fit_list = np.append(fit_list, [fitnesses], axis=0)
            summed_fit_list.append(genome.fitness)
            runtime_list = np.append(runtime_list, [runtime], axis=0)
            time_list = np.append(time_list, [time], axis=0)
            frame_list = np.append(frame_list, [frame], axis=0)
            fitness_weight.append(max(fitnesses[0], 1.0) / max(genome.fitness, 1.0))
            end_frame_list = np.append(end_frame_list, [end_frame], axis=0)

        self.completion_list = completion_list
        self.fit_list = fit_list
        self.bonus_list = bonus_list
        self.time_list = time_list
        self.runtime_list = runtime_list
        self.avg_speed_list = avg_speed_list
        self.avg_completion_list = avg_completion_per_frame_list
        self.summed_fit_list = summed_fit_list
        self.fitness_weight = fitness_weight
        self.autopsy_list = autopsy_list
        self.frame_list = frame_list
        self.end_frame_list = end_frame_list
        print(
            f'=== {datetime.now().strftime("%H:%M:%S")} === Finished Fitness Calculation For > {self.generation}'
        )

    def run(self, num_gens: int = 0):

        winner = self.p.run(self.eval_genomes, num_gens)
        return winner

    def check_eval_status(self, collection):
        for genome in collection.find(
            {
                "generation": self.generation,
                "started_eval": True,
                "finished_eval": False,
                "trial": wandb.config["trial"],
            }
        ):
            genome_id = genome["_id"]
            key = genome["key"]
            started_at = genome["started_at"]
            mins = (np.sum(target_times) * 1.1 + 10.0 * len(target_times)) / 60.0
            if (
                started_at != None
                and (datetime.now() - started_at) > timedelta(minutes=mins)
            ) or ("just_failed" in genome and genome["just_failed"]):
                collection.update_one(
                    {"_id": genome_id},
                    {
                        "$set": {"started_eval": False, "finished_eval": False},
                        "$unset": {"just_failed": ""},
                    },
                )
                if "failed_eval" in genome and genome["failed_eval"]:
                    if "error" in genome and genome["error"] == "Frame rate too low!":
                        hostname = (
                            genome["hostname"]
                            if "hostname" in genome and genome["hostname"]
                            else "Unknown"
                        )
                        frame_rate = genome["frame_rate"]
                        delete_last_lines(6)
                        print(
                            f'=== {datetime.now().strftime("%H:%M:%S")} === Genome {genome["key"]} has a framerate of {frame_rate} on worker {hostname}!'
                        )
                        wandb.alert(
                            title="Low Framerate",
                            text=f"Gen: {self.generation} Genome {genome['key']} on {hostname} had a framerate of {frame_rate}\n\n\n\n\n\n\n",
                        )
                        self.low_framerates += 1
                        continue
                    delete_last_lines(6)
                    print(
                        f'=== {datetime.now().strftime("%H:%M:%S")} === Genome {key} failed to evaluate Worker {genome["hostname"]} Error: {genome["error"] if "error" in genome else "Unknown"}\n\n\n\n\n\n\n'
                    )
                    wandb.alert(
                        title="Failed Eval!",
                        text=f"Gen: {self.generation} Genome {key} failed to evaluate! Worker {genome['hostname']} Error: {genome['error'] if 'error' in genome else 'Unknown'}",
                    )
                    self.failed_evals += 1
                    continue
                delete_last_lines(6)
                print(
                    f'=== {datetime.now().strftime("%H:%M:%S")} === Genome {key} has been running for {round(mins, 1)} minutes, marking for review!\n\n\n\n\n\n\n'
                )
                collection.update_one(
                    {"_id": genome_id},
                    {
                        "$set": {
                            "started_eval": False,
                            "failed_eval": True,
                            "error": "Timeout",
                        }
                    },
                )
                hostname = "Unknown"
                try:
                    if genome["hostname"]:
                        hostname = genome["hostname"]
                except KeyError:
                    pass
                wandb.alert(
                    title="Evaluation Timeout",
                    text=f"Gen: {self.generation} Genome {key} has been running for {round(mins, 1)} minutes, marking for review. Worker {hostname}.",
                )
                self.timedout_evals += 1


manager = EvolveManager(config_path, generation=773)
while True:
    try:
        manager.run(num_gens=1)
        print(
            f'=== {datetime.now().strftime("%H:%M:%S")} === Starting Logging For > {manager.generation}'
        )
        log = {
            "Generation": manager.generation,
            "Time Elapsed": (datetime.now() - manager.gen_start).total_seconds(),
            "Time": datetime.now(),
            "Num Species": len(manager.current_species_list),
            "Population Size": len(manager.fit_list),
            "Num Workers": manager.num_workers,
            "Max Fitness": np.max(manager.summed_fit_list),
            "Min Fitness": np.min(manager.summed_fit_list),
            "Avg Fitness": np.mean(manager.summed_fit_list),
            "Median Fitness": np.median(manager.summed_fit_list),
            "SD Fitness": np.std(manager.summed_fit_list),
            "Avg Combined Completion": np.mean(np.sum(manager.completion_list, axis=1)),
            "Median Combined Completion": np.median(
                np.sum(manager.completion_list, axis=1)
            ),
            "Max Combined Completion": np.max(np.sum(manager.completion_list, axis=1)),
            "Min Combined Completion": np.min(np.sum(manager.completion_list, axis=1)),
            "Avg Fitness Weight": np.mean(manager.fitness_weight),
            "Median Fitness Weight": np.median(manager.fitness_weight),
            "Max Fitness Weight": np.max(manager.fitness_weight),
            "Min Fitness Weight": np.min(manager.fitness_weight),
            "Failed Evaluations": manager.failed_evals,
            "Timedout Evaluations": manager.timedout_evals,
            "Low Framerates": manager.low_framerates,
        }
        for idx, track in enumerate(wandb.config["tracks"]):
            track_log = {
                f"{track} Avg Fitness": np.mean(manager.fit_list[:, idx]),
                f"{track} Median Fitness": np.median(manager.fit_list[:, idx]),
                f"{track} Max Fitness": np.max(manager.fit_list[:, idx]),
                f"{track} Min Fitness": np.min(manager.fit_list[:, idx]),
                f"{track} SD Fitness": np.std(manager.fit_list[:, idx]),
                f"{track} Avg Bonus": np.mean(manager.bonus_list[:, idx]),
                f"{track} Median Bonus": np.median(manager.bonus_list[:, idx]),
                f"{track} Max Bonus": np.max(manager.bonus_list[:, idx]),
                f"{track} Min Bonus": np.min(manager.bonus_list[:, idx]),
                f"{track} SD Bonus": np.std(manager.bonus_list[:, idx]),
                f"{track} Avg Completion": np.mean(manager.completion_list[:, idx]),
                f"{track} Median Completion": np.median(
                    manager.completion_list[:, idx]
                ),
                f"{track} Max Completion": np.max(manager.completion_list[:, idx]),
                f"{track} Min Completion": np.min(manager.completion_list[:, idx]),
                f"{track} SD Completion": np.std(manager.completion_list[:, idx]),
                f"{track} Avg Runtime": np.mean(manager.runtime_list[:, idx]),
                f"{track} Median Runtime": np.median(manager.runtime_list[:, idx]),
                f"{track} Max Runtime": np.max(manager.runtime_list[:, idx]),
                f"{track} Min Runtime": np.min(manager.runtime_list[:, idx]),
                f"{track} SD Runtime": np.std(manager.runtime_list[:, idx]),
                f"{track} Avg Speed": np.mean(manager.avg_speed_list[:, idx]),
                f"{track} Median Speed": np.median(manager.avg_speed_list[:, idx]),
                f"{track} Max Speed": np.max(manager.avg_speed_list[:, idx]),
                f"{track} Min Speed": np.min(manager.avg_speed_list[:, idx]),
                f"{track} SD Speed": np.std(manager.avg_speed_list[:, idx]),
                f"{track} Avg Completion Per Frame": np.mean(
                    manager.avg_completion_list[:, idx]
                ),
                f"{track} Median Completion Per Frame": np.median(
                    manager.avg_completion_list[:, idx]
                ),
                f"{track} Max Completion Per Frame": np.max(
                    manager.avg_completion_list[:, idx]
                ),
                f"{track} Min Completion Per Frame": np.min(
                    manager.avg_completion_list[:, idx]
                ),
                f"{track} SD Completion Per Frame": np.std(
                    manager.avg_completion_list[:, idx]
                ),
                f"{track} Completions": len(
                    manager.completion_list[manager.completion_list[:, idx] == 100.0]
                ),
            }

            for key in manager.autopsy_list[idx].keys():
                track_log[f"{track} {key}"] = manager.autopsy_list[idx][key]
            end_frames = manager.end_frame_list[:, idx]
            end_frames = end_frames.astype(float)
            end_frames = end_frames / 28.0
            time_diffs = manager.runtime_list[:, idx] - end_frames
            track_log[f"{track} Avg Runtime Diff"] = np.mean(time_diffs)
            track_log[f"{track} Median Runtime Diff"] = np.median(time_diffs)
            track_log[f"{track} Max Runtime Diff"] = np.max(time_diffs)
            track_log[f"{track} Min Runtime Diff"] = np.min(time_diffs)
            track_log[f"{track} SD Runtime Diff"] = np.std(time_diffs)

            if np.max(manager.time_list[:, idx]) > 0:
                real_times = manager.time_list[:, idx][manager.time_list[:, idx] > 0]
                track_log[f"{track} Avg Time"] = np.mean(real_times)
                track_log[f"{track} Median Time"] = np.median(real_times)
                track_log[f"{track} Max Time"] = np.max(real_times)
                track_log[f"{track} Min Time"] = np.min(real_times)
                track_log[f"{track} SD Time"] = np.std(real_times)
                real_frames = manager.frame_list[:, idx][manager.time_list[:, idx] > 0]
                ## Convert frames nparray from int to float
                real_frames = real_frames.astype(float)
                real_frames = real_frames / 28.0
                track_log[f"{track} Avg Frame Time"] = np.mean(real_frames)
                track_log[f"{track} Median Frame Time"] = np.median(real_frames)
                track_log[f"{track} Max Frame Time"] = np.max(real_frames)
                track_log[f"{track} Min Frame Time"] = np.min(real_frames)
                track_log[f"{track} SD Frame Time"] = np.std(real_frames)

                if np.shape(real_times) != np.shape(real_frames):
                    print(np.shape(real_times))
                    print(np.shape(real_frames))
                    raise Exception("Time and Frame shape mismatch")

                time_diffs = real_times - real_frames
                track_log[f"{track} Avg Time Diff"] = np.mean(time_diffs)
                track_log[f"{track} Median Time Diff"] = np.median(time_diffs)
                track_log[f"{track} Max Time Diff"] = np.max(time_diffs)
                track_log[f"{track} Min Time Diff"] = np.min(time_diffs)
                track_log[f"{track} SD Time Diff"] = np.std(time_diffs)

            log.update(track_log)
        combined_times = np.sum(manager.time_list, axis=1)[
            np.sum(manager.completion_list, axis=1) == 200.0
        ]
        try:
            combined_time_stats = {
                "Combined Avg Time": np.mean(combined_times),
                "Combined Median Time": np.median(combined_times),
                "Combined Max Time": np.max(combined_times),
                "Combined Min Time": np.min(combined_times),
                "Combined SD Time": np.std(combined_times),
                "Total Completions": len(combined_times),
            }
            log.update(combined_time_stats)
        except:
            pass
        wandb.log(log)
        print(
            f'=== {datetime.now().strftime("%H:%M:%S")} === Finished Logging For > {manager.generation}'
        )

        if not os.path.exists(f"NEAT-Trial_{wandb.config['trial']}"):
            os.makedirs(f"NEAT-Trial_{wandb.config['trial']}")

        for file in os.listdir():
            if (
                file.startswith("NEAT-")
                and not file.startswith("NEAT-Trial")
                and not file.endswith(f"{manager.generation}")
                and not file.endswith(f"{manager.generation - 1}")
            ):
                shutil.move(file, f"NEAT-Trial_{wandb.config['trial']}")
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

if not os.path.exists(f"NEAT-Trial_{wandb.config['trial']}"):
    os.makedirs(f"NEAT-Trial_{wandb.config['trial']}")

for file in os.listdir():
    if (
        file.startswith("NEAT-")
        and not file.startswith("NEAT-Trial")
        and not file.endswith(f"{manager.generation}")
        and not file.endswith(f"{manager.generation - 1}")
    ):
        shutil.move(file, f"NEAT-Trial_{wandb.config['trial']}")
