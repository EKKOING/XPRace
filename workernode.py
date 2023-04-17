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

if socket.gethostname().find(".") >= 0:
    hostname = socket.gethostname()
else:
    hostname = socket.gethostbyaddr(socket.gethostname())[0]

hostname += f"_{args.instance}"
hostname = f"{args.host}_" + hostname

print(f"{args.host} {args.instance} === Beginning Work Cycle ===")
waiting = False
while True:
    with open("pause.txt", "r") as f:
        paused = f.read().strip() == "True"
    if paused:
        if not waiting:
            print(f"{args.host} {args.instance} === Paused ===")
            waiting = True
        sleep(1)
        continue
    if collection.count_documents({"started_eval": False}) != 0:
        genome = collection.find_one_and_update(
            {"started_eval": False},
            update={
                "started_eval": True,
                "hostname": hostname,
                "started_at": datetime.now(),
            },
            sort=[
                ("generation", pymongo.ASCENDING),
                ("individual_num", pymongo.ASCENDING),
            ],
            return_document=pymongo.ReturnDocument.AFTER
        )
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
            runtimes = np.full(num_tracks, -1.0).tolist()
            autopsies = ["Unknown" for _ in range(num_tracks)]
            frames = np.zeros(num_tracks).tolist()
            time_diffs = np.zeros(num_tracks).tolist()
            frame_adj_runtimes = np.full(num_tracks, -1.0).tolist()
            collection.update_one(
                {"_id": genome["_id"]},
                {
                    "$set": {
                        "bonus": bonuses,
                        "completion": completions,
                        "time": times,
                        "runtime": runtimes,
                        "x": last_xs,
                        "y": last_ys,
                        "avg_speed": avg_speeds,
                        "avg_completion_per_frame": avg_completions_per_frame,
                        "frame_rate": 0.0,
                        "finished_eval": False,
                        "autopsy": autopsies,
                        "frame": frames,
                        "end_frame": frames,
                        "time_diff": time_diffs,
                        "frame_adj_runtime": frame_adj_runtimes,
                    }
                },
            )
            client.close()
            print(
                f"{args.host} {args.instance} === Beginning evaluation of genome {individual_num} in generation {generation} on {tracks}!"
            )
            for track_num, track in enumerate(tracks):
                eval_length = 10
                with open(f"{track}.json") as f:
                    track_info = json.load(f)
                    eval_length += track_info["target_time"]
                port_num = randint(49152, 65535)
                print(f"{args.host} {args.instance} === Starting Server on {port_num}! Track: {track}")
                server = subprocess.Popen(
                    [
                        "./xpilots",
                        "-map",
                        f"{track}.xp",
                        "-noQuit",
                        "-maxClientsPerIP",
                        "500",
                        "-password",
                        "test",
                        "-worldlives",
                        "999",
                        "-fps",
                        f"{fps}",
                        "-contactPort",
                        f"{port_num}",
                    ]
                )
                sleep(3)
                print(f"{args.host} {args.instance} === Starting Bot!")
                bot = subprocess.Popen(
                    [
                        "python3",
                        "workerclient.py",
                        "-port",
                        f"{port_num}",
                        "-track",
                        f"{track_num}",
                        "-dbid",
                        f"{str(genome['_id'])}",
                        "-eval_length",
                        f"{eval_length}",
                    ]
                )
                sleep(0.25)
                print(f"{args.host} {args.instance} === Waiting for Bot to finish!")
                bot_return_code = bot.wait()
                sleep(0.25)
                print(f"{args.host} {args.instance} === Bot finished with return code {bot_return_code}!")
                server.terminate()
                sleep(0.25)
                print(f"{args.host} {args.instance} === Server killed!")
                if bot_return_code != 0:
                    raise Exception("Worker Client Error!")
                print(f"{args.host} {args.instance} === Finished Track {track_num + 1}/{len(tracks)} ===")
            client = pymongo.MongoClient(db_string)
            db = client.NEAT
            collection = db.genomes
            collection.update_one(
                {"_id": genome["_id"]}, {"$set": {"finished_eval": True}}
            )
            print(f"{args.host} {args.instance} === Finished Eval Successfully ===")
            waiting = False
            exit(0)
        except Exception as e:
            client = pymongo.MongoClient(db_string)
            db = client.NEAT
            collection = db.genomes
            genome = collection.find_one({"_id": genome["_id"]})
            if genome is None:
                continue
            updates: Dict[str, Any] = {
                "started_eval": True,
                "finished_eval": False,
                "failed_eval": True,
                "just_failed": True,
            }
            if str(e) != "Worker Client Error!":
                updates["exception"] = f"{e}"
                if "error" not in genome:
                    updates["error"] = f"Runtime Exception: {e}"
                raise e
            collection.update_one({"_id": genome["_id"]}, {"$set": updates})
            print(f"{args.host} {args.instance} === Error In Eval: {e}")
            raise e
    else:
        if not waiting:
            print(f"{args.host} {args.instance} === Waiting For Work Assignment!")
            waiting = True
        ## Try to desync workers slightly
        sleep(uniform(1, 5))
    sleep(1)
