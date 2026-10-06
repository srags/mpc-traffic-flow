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