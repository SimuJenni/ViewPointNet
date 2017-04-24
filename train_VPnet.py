from VPNet import VPNet
from datasets.ObjectNet3D import ObjectNet3D
from VPNetTrainer import VPNetTrainer
from Preprocessor import Preprocessor


model = VPNet(num_layers=5, batch_size=128)
data = ObjectNet3D('car')
preprocessor = Preprocessor(target_shape=[96, 96, 3])
trainer = VPNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='first_attempt',
                       lr_policy='const', optimizer='adam')
trainer.train()
