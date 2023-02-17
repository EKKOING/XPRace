import faulthandler
import json
import socket
import subprocess
from datetime import datetime, timedelta
from random import uniform, randint
from time import sleep
import argparse

from typing import List, Union, Dict, Any
import numpy as np
import pymongo

from shellracebot import ShellBot

fps = 28

parser = argparse.ArgumentParser()
parser.add_argument("-instance", help="instance_no", required=True)
parser.add_argument("-host", help="host", required=True)
args = parser.parse_args()

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

hostname = ""
import socket

if socket.gethostname().find('.')>=0:
    hostname = socket.gethostname()
else:
    hostname = socket.gethostbyaddr(socket.gethostname())[0]

hostname += f"_{args.instance}"
hostname = f'{args.host}_' + hostname

print("=== Beginning Work Cycle ===")
waiting = False
while True:
    if collection.count_documents({"started_eval": False}) != 0:
        genome = collection.find_one({"started_eval": False}, sort=[("generation", pymongo.ASCENDING), ("individual_num", pymongo.ASCENDING)])
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
            autopsies = ["Unknown" for _ in range(num_tracks)]
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
                        "frame_rate": 0.0,
                        "hostname": hostname,
                        "finished_eval": False,
                        "autopsy": autopsies
                    }
                },
            )
            client.close()
            print(f'Beginning evaluation of genome {individual_num} in generation {generation} on {tracks}!')
            for track_num, track in enumerate(tracks):
                eval_length = 10
                with open(f'{track}.json') as f:
                    track_info = json.load(f)
                    eval_length += track_info["target_time"]
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
            client = pymongo.MongoClient(db_string)
            db = client.NEAT
            collection = db.genomes
            collection.update_one(
                {"_id": genome["_id"]}, {"$set": {"finished_eval": True}}
            )
            print("=== Finished Eval Successfully ===")
            waiting = False
        except Exception as e:
            client = pymongo.MongoClient(db_string)
            db = client.NEAT
            collection = db.genomes
            updates: "dict[str, Union[str, bool]]" = {"started_eval": False, "finished_eval": False}
            if str(e) != "Bot Error!":
                updates.update({"error": f'{e}'})
            collection.update_one(
                {"_id": genome["_id"]}, {"$set": updates}
            )
            print(f"Error In Eval: {e}")
            print("==================")
            raise e
    else:
        if not waiting:
            print("Waiting For Work Assignment!")
            print("==================")
            waiting = True
        ## Try to desync workers
        sleep(uniform(3, 15))
    sleep(1)
