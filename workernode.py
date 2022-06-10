from shellracebot import ShellBot
from time import sleep
import pymongo
import json
from bson.binary import Binary
import pickle
from datetime import datetime, timedelta
from neat import nn
from GANet import GANet
from random import randint

eval_length = 60

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

print('Starting Bot!')
sb = ShellBot()
sb.start()
# Give Us Enough Time to Type Password In
for idx in range(0, 10):
    print(f'Grant Operator Perms in Next {10 - idx} Seconds!!!')
    sleep(1)
print("=== Beginning Work Cycle ===")
waiting = False
while True:
    try:
        if collection.count_documents({'started_eval': False}) != 0:
            genome = collection.find_one({'started_eval': False})
            collection.update_one({'_id': genome['_id']}, {
                                '$set': {'started_eval': True, 'started_at': datetime.now()}})
            net = pickle.loads(genome['genome'])
            generation = genome['generation']
            individual_num = genome['individual_num']
            sb.nn = net
            sb.reset()
            while sb.awaiting_reset or sb.done:
                pass
            print(
                f'Generation {generation} number {individual_num} started evaluation!')
            start_time = datetime.now()
            while not sb.done and (datetime.now() - start_time).total_seconds() < eval_length:
                sleep(0.01)
            bonus, completion, time = sb.get_scores()
            collection.update_one({'_id': genome['_id']}, {
                                '$set': {'bonus': bonus, 'completion': bonus, 'time': time, 'finished_eval': True}})
            print(f'Generation {generation} number {individual_num} finished evaluation! Bonus: {bonus} Completion: {completion} Time: {time} Runtime: {round((datetime.now() - start_time).total_seconds(), 3)}s')
            print("==================")
            waiting = False
        else:
            if not waiting:
                print('Waiting For Work Assignment!')
                waiting = True
            ## Try to desync workers
            sleep(randint(5, 15))
    except Exception as e:
        print(f'Error: {e}')
        sleep(randint(5, 15))
        print('==================')
    sleep(1)
