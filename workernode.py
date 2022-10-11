import faulthandler
import json
import subprocess
from datetime import datetime, timedelta
from random import randint
from time import sleep

import numpy as np
import pymongo

from shellracebot import ShellBot

eval_length = 120
fps = 28

faulthandler.enable(all_threads=True)

try:
    with open("creds.json") as f:
        creds = json.load(f)
except FileNotFoundError:
    print("creds.json not found!")
    exit()

db_string = creds["mongodb"]
client = pymongo.MongoClient(db_string)
db = client.NEAT
collection = db.genomes

print("=== Beginning Work Cycle ===")
waiting = False
while True:
    try:
        if collection.count_documents({"started_eval": False}) != 0:
            genome = collection.find_one({"started_eval": False})
            if genome is None:
                continue
            try:
                generation = genome["generation"]
                individual_num = genome["individual_num"]
                species = genome["species"]
                tracks = genome["tracks"]
                num_tracks = len(tracks)
                bonuses = np.zeros(num_tracks).tolist()
                completions = np.zeros(num_tracks).tolist()
                times = np.full(num_tracks, -1.0).tolist()
                last_xs = np.zeros(num_tracks).tolist()
                last_ys = np.zeros(num_tracks).tolist()
                avg_speeds = np.zeros(num_tracks).tolist()
                avg_completions_per_frame = np.zeros(num_tracks).tolist()
                runtimes = np.zeros(num_tracks).tolist()
                collection.update_one(
                    {"_id": genome["_id"]},
                    {
                        "$set": {
                            "started_eval": True,
                            "started_at": datetime.now(),
                            "bonus": bonuses,
                            "completion": completions,
                            "time": times,
                            "runtime": runtimes,
                            "x": last_xs,
                            "y": last_ys,
                            "avg_speed": avg_speeds,
                            "avg_completion_per_frame": avg_completions_per_frame,
                            "frame_rate": 0.0
                        }
                    },
                )
                print(f'Beginning evaluation of genome {individual_num} in generation {generation} on {tracks}!')
                for track_num, track in enumerate(tracks):
                    port_num = randint(49152, 65535)
                    print(f"Starting Server on {port_num}! Track: {track}")
                    server = subprocess.Popen(
                        [
                            "./xpilots",
                            "-map", f"{track}.xp",
                            "-noQuit",
                            "-maxClientsPerIP", "500",
                            "-password", "test",
                            "-worldlives", "999",
                            "-fps", f"{fps}",
                            "-contactPort", f"{port_num}",
                        ]
                    )
                    sleep(3)
                    print("Starting Bot!")
                    bot = subprocess.Popen(
                        [
                            "python3",
                            "workerclient.py",
                            "-port", f"{port_num}",
                            "-track", f"{track_num}",
                            "-dbid", f"{str(genome['_id'])}",
                            "-eval_length", f"{eval_length}",
                        ]
                    )
                    sleep(1)
                    print("Waiting for Bot to finish!")
                    bot_return_code = bot.wait()
                    sleep(1)
                    print(f"Bot finished with return code {bot_return_code}!")
                    if bot_return_code != 0:
                        server.kill()
                        server.wait()
                        print("Server killed!")
                        raise Exception("Bot Error!")
                    server.kill()
                    sleep(1)
                    print("Server killed!")
                    print(f"=== Finished Track {track_num + 1}/{len(tracks)} ===")
                collection.update_one(
                    {"_id": genome["_id"]}, {"$set": {"finished_eval": True}}
                )
                print("=== Finished Eval Successfully ===")
                waiting = False
            except Exception as e:
                collection.update_one(
                    {"_id": genome["_id"]}, {"$set": {"started_eval": False}}
                )
                print(f"Error In Eval: {e}")
                waiting = False
                print("==================")
        else:
            if not waiting:
                print("Waiting For Work Assignment!")
                print("==================")
                waiting = True
            ## Try to desync workers
            sleep(randint(5, 15))
    except Exception as e:
        print(f"Error: {e}")
        sleep(randint(5, 15))
        print("==================")
    sleep(1)
