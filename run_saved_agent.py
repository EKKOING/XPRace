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

genome = None
human = False
print("Enter the trial number, generation number, and individual number of the genome you want to run. Leave blank to run the best genome according to criteria. Type 'Human' to run a human player.")
trial = input("Trial Number: ")
if trial == "Human":
    genome = {
        "trial": 0,
        "generation": 0,
        "individual_num": 0,
        "species": 0,
        "_id": "Human",
        "tracks": ["circuit1_a", "circuit1_b"],
    }
    human = True
    print("Running Human Trial!")
else:
    if trial == "":
        genome = collection.find_one({}, sort=[("fitness", pymongo.DESCENDING)])
    else:
        trial = int(trial)
        generation = input("Generation: ")
        if generation == "":
            genome = collection.find_one({"trial": trial}, sort=[("fitness", pymongo.DESCENDING)])
        else:
            generation = int(generation)
            individual_num = input("Individual Number: ")
            if individual_num == "":
                genome = collection.find_one({"trial": trial, "generation": generation}, sort=[("fitness", pymongo.DESCENDING)])
            else:
                individual_num = int(individual_num)
                genome = collection.find_one({"trial": trial, "generation": generation, 'individual_num': individual_num})
    if genome is None:
        print("Genome not found!")
        exit()
generation = genome["generation"]
individual_num = genome["individual_num"]
species = genome["species"]
tracks = genome["tracks"]
trial = genome["trial"]
print(f'Running genome {individual_num} in generation {generation} from trial {trial} on {tracks}!')
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
            "testclient.py",
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
print("=== Finished Test Successfully ===")
