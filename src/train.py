import json
import os

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from utils import register_dataset, get_custom_config
import utils
import matplotlib.pyplot as plt

register_dataset(split='train')
train_metadata, dataset_dicts = MetadataCatalog.get('coco-train'), DatasetCatalog.get('coco-train')

register_dataset(split='test')

cfg = get_custom_config(load_saved=False)

if os.path.exists(cfg.OUTPUT_DIR + '/metrics.json'):
    os.remove(cfg.OUTPUT_DIR + '/metrics.json')

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Load the metrics saved during training to visualize loss
def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines
experiment_metrics = load_json_arr(cfg.OUTPUT_DIR + '/metrics.json')

plt.rcParams['figure.figsize'] = [15, 8]
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()

print('''

{}Training Successful!{}

Initial model was from: {}

Saved final model to: {}

'''.format(utils.OKGREEN, utils.ENDC, 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')))