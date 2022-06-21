from time import sleep
import pymongo
import json
import pickle
from neat import nn
from GANet import GANet
from datetime import datetime, timedelta

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

generation = input("Generation Number: ")
try:
    generation = int(generation)
except ValueError:
    print('Generation must be an integer!')
    exit()

trial = input("Trial: ")
try:
    trial = float(trial)
except ValueError:
    trial = int(trial)
except Exception:
    print('Trial must be an integer or float!')
    exit()

num_agents = collection.count_documents({'generation': generation, 'trial': trial, 'genome': {'$ne': None}})
while num_agents == 0:
    generation += 1
    print(f'Searching for {trial}-{generation}!')
    num_agents = collection.count_documents({'generation': generation, 'trial': trial, 'genome': {'$ne': None}})

print(f'{num_agents} agents found in generation {generation}!')

from shellracebot import ShellBot
print('Starting Bot!')
sb = ShellBot()
sb.start()
# Give Us Enough Time to Type Password In
for idx in range(0, 10):
    print(f'Grant Operator Perms in Next {10 - idx} Seconds!!!')
    sleep(1)
eval_length = 120.0
print('==== RUN START ====')
while True:
    try:
        for genome in collection.find({'generation': generation, 'trial': trial, 'genome': {'$ne': None}}).sort([('completion', pymongo.DESCENDING), ('time', pymongo.ASCENDING)]):
            print(f'Using {trial} {genome["generation"]}-{genome["individual_num"]}!')
            sb.nn = pickle.loads(genome['genome'])
            sb.reset()
            while sb.done or sb.awaiting_reset:
                pass
            start_time = datetime.now()
            while not sb.done and (datetime.now() - start_time).total_seconds() < eval_length:
                sleep(0.01)
            bonus, completion, time = sb.get_scores()
            run_time = round((datetime.now() - start_time).total_seconds(), 3)
            print(f'Bonus: {round(bonus, 3)}, Completion: {round(completion, 3)}, Time: {round(time, 3)}s, Runtime: {run_time}s')
    except KeyboardInterrupt:
        print('==== Exit Request Received! ====')
        break
