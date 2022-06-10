import libpyAI as ai

def run():
    x = ai.selfX()
    y = ai.selfY()
    print(f'[{x}, {y}]')

ai.start(run, ["-name", "Coord Bot", "-join", "localhost"])