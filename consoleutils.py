import sys
# Output Utils
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


def delete_last_lines(n: int = 1) -> None:
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

def get_bar_graph(percent: float, width: int = 21, center: bool = False) -> str:
    percent = min(1.0, max(-1.0, percent))
    num_bars = int(round(abs(percent) * float(width)))
    bar = '█'
    graph = '['
    if center:
        num_bars = int(round(num_bars / 2.0))
        half_width = int(round(width / 2))
        if num_bars == 0:
            graph += ' ' * half_width
            graph += '|'
            graph += ' ' * half_width
        elif percent > 0:
            for _ in range(half_width):
                graph += ' '
            graph += '|'
            for _ in range(num_bars):
                graph += bar
            for _ in range(half_width - num_bars):
                graph += ' '
        else:
            for _ in range(half_width - num_bars):
                graph += ' '
            for _ in range(num_bars):
                graph += bar
            graph += '|'
            for _ in range(half_width):
                graph += ' '
    else:
        percent = max(0.0, percent)
        for _ in range(int(num_bars)):
            graph += '█'
        for _ in range(int(width - num_bars)):
            graph += ' '
    graph += ']'
    return graph

def progress_bar(progress: float, in_progress: float, total: float) -> None:
    left = '['
    right = ']'
    bar_length = 30
    fill = '|'
    in_progress_fill = '>'
    percent = progress / total
    in_percent = in_progress / total
    fill_amt = int(round(percent * bar_length))
    fill_str = ''
    for _ in range(fill_amt):
        fill_str += fill
    in_fill_amt = int(round(in_percent * bar_length))
    for _ in range(in_fill_amt):
        fill_str += in_progress_fill
    for _ in range(bar_length - (fill_amt + in_fill_amt)):
        fill_str += ' '
    print(f'{left}{fill_str}{right} {percent:.2%}')
