# Load the metrics saved during training to visualize loss
import json

from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 2:
    print('Usage: python plot_loss.py <metrics.json>')
    sys.exit()


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


experiment_metrics = load_json_arr(sys.argv[1])

plt.rcParams['figure.figsize'] = [15, 8]
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' not in x],
    [x['total_loss'] for x in experiment_metrics if 'bbox/AP' not in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x],
    [x['total_loss'] for x in experiment_metrics if 'bbox/AP' in x])
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()
