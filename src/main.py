import pytorch_lightning as pl
from model.CnnBirdDetector import CnnBirdDetector
from data_module.sample_module import SampleModule

# nd without changing a single line of code, you could run on GPUs/TPUs
# 8 GPUs
# trainer = Trainer(max_epochs=1, gpus=8)
# 256 GPUs
# trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32)
# TPUs
# trainer = Trainer(tpu_cores=8)
data_module = SampleModule()
model = CnnBirdDetector()
trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=20)
trainer.fit(model, data_module)
