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

try:
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
    track_num = None
    if args.track.isnumeric():
        track_num = int(args.track)
    else:
        track = args.track
    if not args.dbid:
        print("No genome id specified!")
        exit(2)
    print(f'port {args.port} for track {track_num} with genome id {args.dbid}')
    if args.dbid == "Human":
        db_objid = "Human"
    else:
        db_objid = ObjectId(args.dbid)
    if not args.eval_length:
        print("No evaluation length specified!")
        exit(2)
    eval_length = float(args.eval_length)

    ## Start bot
    try:
        with open('creds.json') as f:
            creds = json.load(f)
    except FileNotFoundError:
        print('creds.json not found!')
        exit(1)

    db_string = creds['mongodb']
    client = pymongo.MongoClient(db_string)
    db = client.NEAT
    collection = db.genomes
    generation = 0
    individual_num = 0
    species = 0
    tracks = ["circuit1_a", "circuit1_b"]
    track = tracks[track_num]
    net = None
    if db_objid != "Human":
        genome = collection.find_one({'_id': db_objid})
        if genome is None:
            print('Genome invalid!')
            exit(2)
        net = pickle.loads(genome['genome'])
        generation = genome['generation']
        individual_num = genome['individual_num']
        species = genome['species']
        tracks = genome['tracks']
        if track_num is not None:
            track = tracks[track_num]
    else:
        human = True
    sb = ShellBot(f"EKKO{track_num}", track, args.port, headless=False, human = human)
    sb.start()
    sleep(1)
    sb.ask_for_perms = True
    sleep(1)
    if db_objid != "Human":
        sb.nn = net
    sb.reset()
    while sb.awaiting_reset or sb.done:
        pass
    print(f'Generation {generation} number {individual_num} started run on {track}!')
    start_time = datetime.now()
    sb.show_info = True
    while not sb.done and (datetime.now() - start_time).total_seconds() < eval_length:
        sleep(0.01)
    if not sb.done and (datetime.now() - start_time).total_seconds() >= eval_length:
        sb.cause_of_death = "Time"
    sb.show_info = False
    bonus, completion, time = sb.get_scores()
    frame_rate = sb.frame_rate
    avg_speed = round(sb.average_speed, 3)
    avg_completion_per_frame = round(sb.average_completion_per_frame, 3)
    runtime = round((datetime.now() - start_time).total_seconds(), 3)
    try:
        sb.close_bot()
        print('Bot Closed!')
    except Exception:
        pass
    print(f'Generation {generation} number {individual_num} Species: {species} finished run on {track}! Bonus: {round(bonus, 3)} Completion: {round(completion, 3)} Time: {time} Runtime: {runtime}s Avg Speed: {avg_speed} Avg Completion: {avg_completion_per_frame}')
    print(f'Frame Rate: {frame_rate}')
    print(f'Cause of Death: {sb.cause_of_death}')
except Exception as e:
    print('Error in workerclient.py')
    raise e
exit(0)
