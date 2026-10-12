from pathlib import Path
from tabulate import tabulate
from .paths import cut_repo

colors = {
  'red': '\033[91m',
  'green': '\033[92m',
  'yellow': '\033[93m',
  'blue': '\033[94m',
  'magenta': '\033[95m',
  'cyan': '\033[96m',
  'white': '\033[97m',
  'bold': '\033[1m',
  'underline': '\033[4m',
  'end': '\033[0m'
}

def colored_helper(text, color): 
  return f"{colors.get(color, '')}{text}{colors['end']}"
def colored(text, *colors):
  for c in colors: text = colored_helper(text, c)
  return text

def print_save(file_path: Path):
  print(f"Saved to {colored(cut_repo(file_path), 'green', 'bold')}")

def print_title(text, color='yellow', bold=True):
  print(colored(f"<< {text} >>", color, 'bold' if bold else ''))

def print_file(file_path: Path):
  text = f"RUNNING: {cut_repo(file_path)}"
  line = colored('='*len(text), 'cyan', 'bold')
  print(line); print(colored(text, 'cyan', 'bold')); print(line)

def print_table(data: list[list], headers: list = [], tablefmt="outline"):
  print(tabulate(data, headers=headers, tablefmt=tablefmt))