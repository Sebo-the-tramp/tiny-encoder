import time
start_tm = time.perf_counter()

import math
from typing import Tuple, cast
import numpy as np
from tinygrad import Tensor, GlobalCounters, TinyJit, Context
from tinygrad.helpers import getenv, Context
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from utils.data_generation import generate_dataset
from utils.layout import TrainingDashboard
from encoders.mlp_encoder_256 import MLPEncoder256

# Create logs directory if it doesn't exist
log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter(log_dir)

batches = 100

batchsize = getenv("BS", 1024*batches)
bias_scaler = 64
hyp = {
  'misc': {   
    'train_epochs': 10000,
    'device': 'cuda',
  }
}

if __name__ == "__main__":        

    # *** dataset ***
    X_train, X_val, X_test = generate_dataset(batch_size=1024, n_batches=batches, data_type='random')
    print(f"X_train: {X_train.shape} X_val: {X_val.shape} X_test: {X_test.shape}")

    # *** model ***
    model = MLPEncoder256()

    dashboard = TrainingDashboard(epochs=hyp['misc']['train_epochs'])    

    num_steps_per_epoch          = X_train.shape[0] // (1024*batches)
    total_train_steps            = math.ceil(num_steps_per_epoch * hyp['misc']['train_epochs'])
    loss_batchsize_scaler        = 512/1024*batches

    @TinyJit
    @Tensor.train()
    def train_step(idxs:Tensor) -> Tensor:
        with Context(SPLIT_REDUCEOP=0, FUSE_ARANGE=1):
            # Get the batch using tensor indexing
            X = X_train[idxs]
            
            # Forward pass
            out = model(X)
            loss = model.loss_fn(out, X)
            model.opt.zero_grad()
            loss.backward()
            model.opt.step()
            # model.lr_sched.step(current=loss.item())
            return loss / (batchsize*loss_batchsize_scaler)

    eval_batchsize = 1024
    @TinyJit
    @Tensor.test()
    def val_step() -> Tuple[Tensor, Tensor]:
        Tensor.no_grad = True
        loss = []
        for i in range(0, X_val.shape[0], eval_batchsize):
            out = model(X_val)
            loss.append(model.loss_fn(out, X_val))
        ret = Tensor.stack(*loss).mean() / (batchsize*loss_batchsize_scaler)
        Tensor.no_grad = False
        return ret
    
    @TinyJit
    @Tensor.test()
    def test_step() -> Tuple[Tensor, Tensor]:
        Tensor.no_grad = True
        loss = []
        for i in range(0, X_test.shape[0], eval_batchsize):
            out = model(X_test)
            loss.append(model.loss_fn(out, X_test))
        ret = Tensor.stack(*loss).mean() / (batchsize*loss_batchsize_scaler)
        Tensor.no_grad = False
        return ret

    np.random.seed(1337)
    with Context(BEAM=2):
        for epoch in range(math.ceil(hyp['misc']['train_epochs'])):
            gst = time.perf_counter()
            idxs = np.arange(X_train.shape[0])
            np.random.shuffle(idxs)
            tidxs = Tensor(idxs, dtype='int')[:num_steps_per_epoch*1024*80].reshape(num_steps_per_epoch, 1024*80)
            train_loss:float = 0
            curr_gflops = 0
            for epoch_step in (t:=range(num_steps_per_epoch)):
                st = time.perf_counter()
                GlobalCounters.reset()
                loss = train_step(tidxs[epoch_step].contiguous()).float().item()
                current_lr = model.opt.lr.item()                
                current_gflops = GlobalCounters.global_ops / (1e9 * (time.perf_counter() - st))
                train_loss += loss

            train_loss /= num_steps_per_epoch

            gmt = time.perf_counter()
            GlobalCounters.reset()
            val_loss = val_step().float().item()
            get = time.perf_counter()            
            
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)
            writer.add_scalar('Performance/GFLOPS', current_gflops, epoch)
            
            if(epoch % 100 == 0):            
                dashboard.update(step=0, loss=train_loss, learning_rate=current_lr, epoch=epoch, gflops=current_gflops)
                
            if(epoch % 300 == 0):
                model.opt.lr.assign(model.opt.lr * 0.95).realize()

    test_loss = test_step().float().item()
    dashboard.close()
    writer.close()  # Close TensorBoard writer
    print("Finished training")
    print(f"*** test_loss: {test_loss:5.10e} @ {time.perf_counter()-start_tm:6.2f} s  ")