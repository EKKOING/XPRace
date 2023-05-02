import argparse
import json
import pickle
from datetime import datetime
from time import sleep

import pymongo
from bson.objectid import ObjectId
from bson.binary import Binary
from neat import nn

from shellracebot import ShellBot

## Get port number track name and bot name from command line
parser = argparse.ArgumentParser()
parser.add_argument("-port", help="port number", required=True)
parser.add_argument("-track", help="track idx", required=True)
parser.add_argument("-dbid", help="genome db id", required=True)
parser.add_argument("-eval_length", help="evaluation length", required=True)
args = parser.parse_args()

if not args.port:
    print("No port number specified!")
    exit(2)
if not args.track:
    print("No track specified!")
    exit(2)
track_num = int(args.track)
if not args.dbid:
    print("No genome id specified!")
    exit(2)
print(f"port {args.port} for track {track_num} with genome id {args.dbid}")
db_objid = ObjectId(args.dbid)
if not args.eval_length:
    print("No evaluation length specified!")
    exit(2)
try:
    with open("creds.json") as f:
        creds = json.load(f)
except FileNotFoundError:
    print("creds.json not found!")
    exit(1)

db_string = creds["mongodb"]
client = pymongo.MongoClient(db_string)
db = client.NEAT
collection = db.genomes

try:
    genome = collection.find_one({"_id": db_objid})
    if genome is None:
        print("Genome invalid!")
        exit(2)
    eval_length = float(args.eval_length)
    net = pickle.loads(genome["genome"])
    generation = genome["generation"]
    individual_num = genome["individual_num"]
    species = genome["species"]
    tracks = genome["tracks"]
    track = tracks[track_num]
    hostname = genome["hostname"]
    host = hostname.split("_")[0]
    instance = hostname.split("_")[-1]

    sb = ShellBot(f"EKKO{track_num}", track, args.port, headless=True, adv_log=True)
    sb.start()
    sleep(1)
    sb.ask_for_perms = True
    sleep(1)
    sb.nn = net
    sb.reset()
    while sb.awaiting_reset or sb.done:
        pass
    print(
        f"{host} {instance} === Generation {generation} number {individual_num} started logging on {track}!"
    )
    start_time = datetime.now()
    sb.show_info = True
    while not sb.done and (datetime.now() - start_time).total_seconds() < eval_length:
        sleep(0.01)
    if not sb.done and (datetime.now() - start_time).total_seconds() >= eval_length:
        sb.cause_of_death = "Time"
    xs = sb.xs
    ys = sb.ys
    headings = sb.headings
    sb.show_info = False

    frame_rate = 0.0
    if sb.frame_rate:
        frame_rate = sb.frame_rate

    if sb.frame == 0:
        collection.update_one(
            {"_id": genome["_id"]},
            {"$set": {"needs_adv_logging": True}},
        )
        exit(1)
    if frame_rate < 27.9:
        collection.update_one(
            {"_id": genome["_id"]},
            {"$set": {"needs_adv_logging": True}},
        )
        exit(1)
    collection.update_one(
        {"_id": genome["_id"]},
        {
            "$set": {
                f"{track}_xs": xs,
                f"{track}_ys": ys,
                f"{track}_headings": headings
            }
        },
    )
    try:
        sb.close_bot()
        print(f"{host} {instance} === Bot Closed!")
    except Exception:
        pass
    print(
        f"{host} {instance} === Generation {generation} number {individual_num} Species: {species} finished logging on {track}!"
    )
    print(f"{host} {instance} === Frame Rate: {frame_rate}")
except Exception as e:
    print("Error in loggerclient.py")
    genome = collection.find_one({"_id": db_objid})
    if genome is None:
        print("Genome invalid!")
        exit(2)
    update_dict = {
        "needs_adv_log": True,
    }
    collection.update_one({"_id": genome["_id"]}, {"$set": update_dict})
    raise e
exit(0)
