import json
import pickle
import subprocess
from datetime import datetime, timedelta
from random import randint
from time import sleep

import numpy as np
import pymongo
from bson.binary import Binary
from neat import nn

from shellracebot import ShellBot

eval_length = 120

try:
    with open('creds.json') as f:
        creds = json.load(f)
except FileNotFoundError:
    print('creds.json not found!')
    exit()

db_string = creds['mongodb']
client = pymongo.MongoClient(db_string)
db = client.NEAT
collection = db.genomes

print("=== Beginning Work Cycle ===")
waiting = False
while True:
    try:
        if collection.count_documents({'started_eval': False}) != 0:
            genome = collection.find_one({'started_eval': False})
            if genome is None:
                continue
            collection.update_one({'_id': genome['_id']}, {
                                '$set': {'started_eval': True, 'started_at': datetime.now()}})
            net = pickle.loads(genome['genome'])
            generation = genome['generation']
            individual_num = genome['individual_num']
            species = genome['species']
            tracks = genome['tracks']
            num_tracks = len(tracks)
            bonuses = np.zeros(num_tracks).tolist()
            completions = np.zeros(num_tracks).tolist()
            times = np.full(num_tracks, -1.0).tolist()
            last_xs = np.zeros(num_tracks).tolist()
            last_ys = np.zeros(num_tracks).tolist()
            avg_speeds = np.zeros(num_tracks).tolist()
            avg_completions_per_frame = np.zeros(num_tracks).tolist()
            runtimes = np.zeros(num_tracks).tolist()
            for track_num, track in enumerate(tracks):
                print('Starting Server!')
                server = subprocess.Popen(['./xpilots', '-map', f'{track}.xp', '-noQuit', '-maxClientsPerIP', '500', '-password', 'test', '-worldlives', '999', '-reset', 'no', '-fps', '28'])
                print('Starting Bot!')
                sb = ShellBot("EKKO", track)
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
                while not sb.done and (datetime.now() - start_time).total_seconds() < eval_length:
                    sleep(0.01)
                    ## Early Termination
                    if (datetime.now() - start_time).total_seconds() > 10.0 and sb.y < 100:
                        break
                bonuses[track_num], completions[track_num], times[track_num] = sb.get_scores()
                last_xs[track_num] = sb.x
                last_ys[track_num] = sb.y
                avg_speeds[track_num] = round(sb.average_speed, 3)
                avg_completions_per_frame[track_num] = round(sb.average_completion_per_frame, 3)
                runtimes[track_num] = round((datetime.now() - start_time).total_seconds(), 3)
                try:
                    server.terminate()
                except Exception:
                    pass
                print(f'Generation {generation} number {individual_num} Species: {species} finished evaluation on {track}! Bonus: {round(bonuses[track_num], 3)} Completion: {round(completions[track_num], 3)} Time: {times[track_num]} Runtime: {runtimes[track_num]}s Avg Speed: {avg_speeds[track_num]} Avg Completion: {avg_completions_per_frame[track_num]}')
                print("==================")
            collection.update_one({'_id': genome['_id']}, {
                '$set': {'bonus': bonuses, 'completion': completions, 'time': times, 'finished_eval': True, 'runtime': runtimes, 'x': last_xs, 'y': last_ys, 'avg_speed': avg_speeds, 'avg_completion_per_frame': avg_completions_per_frame}})
            waiting = False
        else:
            if not waiting:
                print('Waiting For Work Assignment!')
                print("==================")
                waiting = True
            ## Try to desync workers
            sleep(randint(5, 15))
    except Exception as e:
        print(f'Error: {e}')
        sleep(randint(5, 15))
        print('==================')
    sleep(1)
