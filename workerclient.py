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
    track_num = int(args.track)
    if not args.dbid:
        print("No genome id specified!")
        exit(2)
    print(f'port {args.port} for track {track_num} with genome id {args.dbid}')
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

    genome = collection.find_one({'_id': db_objid})
    if genome is None:
        print('Genome invalid!')
        exit(2)
    net = pickle.loads(genome['genome'])
    generation = genome['generation']
    individual_num = genome['individual_num']
    species = genome['species']
    tracks = genome['tracks']
    track = tracks[track_num]
    bonuses = genome['bonus']
    completions = genome['completion']
    times = genome['time']
    last_xs = genome['x']
    last_ys = genome['y']
    avg_speeds = genome['avg_speed']
    avg_completions_per_frame = genome['avg_completion_per_frame']
    runtimes = genome['runtime']

    sb = ShellBot(f"EKKO{track_num}", track, args.port, headless=True)
    sb.start()
    sleep(1)
    sb.ask_for_perms = True
    sleep(1)
    sb.nn = net
    sb.reset()
    while sb.awaiting_reset or sb.done:
        pass
    print(f'Generation {generation} number {individual_num} started evaluation on {track}!')
    start_time = datetime.now()
    sb.show_info = True
    while not sb.done and (datetime.now() - start_time).total_seconds() < eval_length:
        sleep(0.01)
        ## Early Termination
        if (datetime.now() - start_time).total_seconds() > 10.0 and sb.y < 100:
            break
    sb.show_info = False
    bonuses[track_num], completions[track_num], times[track_num] = sb.get_scores()
    last_xs[track_num] = sb.x
    last_ys[track_num] = sb.y
    avg_speeds[track_num] = round(sb.average_speed, 3)
    avg_completions_per_frame[track_num] = round(sb.average_completion_per_frame, 3)
    runtimes[track_num] = round((datetime.now() - start_time).total_seconds(), 3)
    collection.update_one({'_id': genome['_id']}, {
    '$set': {'bonus': bonuses, 'completion': completions, 'time': times, 'runtime': runtimes, 'x': last_xs, 'y': last_ys, 'avg_speed': avg_speeds, 'avg_completion_per_frame': avg_completions_per_frame}})
    try:
        sb.close_bot()
        print('Bot Closed!')
    except Exception:
        pass
    print(f'Generation {generation} number {individual_num} Species: {species} finished evaluation on {track}! Bonus: {round(bonuses[track_num], 3)} Completion: {round(completions[track_num], 3)} Time: {times[track_num]} Runtime: {runtimes[track_num]}s Avg Speed: {avg_speeds[track_num]} Avg Completion: {avg_completions_per_frame[track_num]}')
except Exception as e:
    print('Error in workerclient.py')
    raise e
exit(0)
