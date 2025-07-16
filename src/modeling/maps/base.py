
# Third-party modules
import torch
import numpy as np
from tqdm import tqdm

# Custom modules
from src.entities.ppi import PPI
from src.misc.logger import logger
from src.modeling.utils import modelhub
from src.modeling.utils.split import Split
from src.modeling.utils.tracker import Tracker
from src.modeling.utils.train import train, evaluate
from src.modeling.utils.performance import Performance
from src.modeling.utils.early_stop import EarlyStopping
from src.modeling.utils.custom_dataset import CustomDataset
logger.setLevel(20)

# Config file
config = {
    'seed': 42,
    'batch_size': 32,

    'train_size': 0.75,
    'val_size': 0.25,
    'test_size': 0.0,
    'kfold': 0,

    'lr': 0.001,
    'epochs': 200,
    'weight_decay': 0,
    'early_stopping_patience': 4,
    'early_stopping_min_delta': 0.0001,
}

# Set seed
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

# GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')
logger.info(f'CUDA available: {torch.cuda.is_available()}')
logger.info(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        logger.info(f'GPU {i}: {torch.cuda.get_device_name(i)}')

# Load and pad dataset
ppis = [ppi for ppi in PPI.iterate('distance_map.CA') if ppi.interact() != '?']
cmaps = [ppi.distance_map['CA'].cmap(8) for ppi in tqdm(ppis, desc='Loading PPI maps')]
max_size = max(map(len, cmaps))
logger.info(f'Maximum map size for padding reference: {max_size}')
X = [cmap.pad(max_size).matrix for cmap in cmaps]
X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
y = [ppi.interact() for ppi in ppis]
y = torch.tensor(np.array(y), dtype=torch.long)
input_dataset = CustomDataset(y=y, cmap=X)
data_loader = torch.utils.data.DataLoader(input_dataset, batch_size=config['batch_size'], shuffle=True)

# Split dataset
split = Split(
    data_loader = data_loader,
    batch_size = config['batch_size'],
    sizes = [config['train_size'], config['val_size'], config['test_size']],
    kfold = 0
)

# Model
cnn = modelhub.CCL(max_shape=max_size)
cnn.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    cnn.parameters(), 
    lr = config['lr'], 
    weight_decay = config['weight_decay'])
early_stopping = EarlyStopping(
    patience=config['early_stopping_patience'],
    min_delta=config['early_stopping_min_delta']
    ) 

# Train + evaluate
tracker = Tracker()
for epoch in tqdm(range(config['epochs'])):
    # Train
    train(model=cnn, train_loader=split.train_loader, criterion=criterion, optimizer=optimizer, device=device)
    # Evaluate on training set
    train_results = evaluate(model=cnn, loader=split.train_loader, criterion=criterion, device=device)
    # Evaluate on test set
    test_results = evaluate(model=cnn, loader=split.val_loader, criterion=criterion, device=device)
    # Performance
    train_performance = Performance(true=train_results['labels'], logits=train_results['logits'])
    test_performance = Performance(true=test_results['labels'], logits=test_results['logits'])
    # Logging
    logger.info(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_results['loss']}, Test Loss: {test_results['loss']}, Train BalAccuracy: {train_performance.balanced_accuracy}, Test BalAccuracy: {test_performance.balanced_accuracy}")
    # Track
    tracker.track(
        #learning_rate = scheduler.get_last_lr(),
        train_loss = train_results['loss'],
        val_loss = test_results['loss'],
        train_bal_accuracy = train_performance.balanced_accuracy,
        val_bal_accuracy = test_performance.balanced_accuracy, 
        train_f1 = train_performance.f1,
        val_f1 = test_performance.f1,
        train_mcc = train_performance.mcc,
        val_mcc = test_performance.mcc
    )
    # Early stopping
    early_stopping(test_results['loss'])
    if early_stopping.early_stop:
        logger.info("Early stopping triggered!")
        break

# Report training and performance
best_epoch, best_metrics = tracker.best_epoch()

# Plot
tracker.plot()
test_performance.plot_confusion_matrix()
test_performance.plot_roc_curve()
test_performance.plot_calibration_curve(n_bins=15, strategy='quantile', model_name='CNN')