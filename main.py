import time
start_tm = time.perf_counter()

import math
from typing import Tuple, cast
import numpy as np
from tinygrad import Tensor, nn, GlobalCounters, TinyJit, dtypes
from tinygrad.helpers import trange, getenv, Context
from extra.lr_scheduler import CosineAnnealingLR

from utils.data_generation import generate_dataset
from encoders.mlp_encoder import MLPEncoder

batchsize = getenv("BS", 1024*80)
bias_scaler = 64
hyp = {
  'misc': {   
    'train_epochs': 1000,
    'device': 'mps',
  },
  'opt': {
    'loss_scale_scaler': 1./32, # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
  },
}

if __name__ == "__main__":
    # *** dataset ***
    X_train, X_val, X_test = generate_dataset(batch_size=1024, n_batches=80, data_type='random')
    print(f"X_train: {X_train.shape} X_val: {X_val.shape} X_test: {X_test.shape}")

    # *** model ***
    model = MLPEncoder()
    state_dict = nn.state.get_state_dict(model)

    num_steps_per_epoch          = X_train.shape[0] // batchsize
    total_train_steps            = math.ceil(num_steps_per_epoch * hyp['misc']['train_epochs'])
    loss_batchsize_scaler        = 512/batchsize

    opt                          = nn.optim.Adam(nn.state.get_parameters(model), lr=0.01)
    lr_sched                     = CosineAnnealingLR(opt, T_max=total_train_steps, eta_min=0.00000001)

    def loss_fn(out, Y):
        return ((out - Y) ** 2).mean()
        # return ((out - Y) ** 2).mul(hyp['opt']['loss_scale_scaler']*loss_batchsize_scaler).sum().div(hyp['opt']['loss_scale_scaler'])

    @TinyJit
    @Tensor.train()
    def train_step(idxs:Tensor) -> Tensor:
        with Context(SPLIT_REDUCEOP=0, FUSE_ARANGE=1):
            # Get the batch using tensor indexing
            X = X_train[idxs]
            
            # Forward pass
            out = model(X)
            loss = loss_fn(out, X)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()
            return loss / (batchsize*loss_batchsize_scaler)

    eval_batchsize = 1024
    @TinyJit
    @Tensor.test()
    def val_step() -> Tuple[Tensor, Tensor]:
        Tensor.no_grad = True
        loss = []
        for i in range(0, X_val.shape[0], eval_batchsize):
            out = model(X_val)
            loss.append(loss_fn(out, X_val))
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
            loss.append(loss_fn(out, X_test))
        ret = Tensor.stack(*loss).mean() / (batchsize*loss_batchsize_scaler)
        Tensor.no_grad = False
        return ret

    np.random.seed(1337)
    for epoch in range(math.ceil(hyp['misc']['train_epochs'])):
        gst = time.perf_counter()
        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)
        tidxs = Tensor(idxs, dtype='int')[:num_steps_per_epoch*batchsize].reshape(num_steps_per_epoch, batchsize)
        train_loss:float = 0
        curr_gflops = 0
        for epoch_step in (t:=trange(num_steps_per_epoch)):
            st = time.perf_counter()
            GlobalCounters.reset()
            loss = train_step(tidxs[epoch_step].contiguous()).float().item()
            current_lr = lr_sched.get_lr().item()
            current_gflops = GlobalCounters.global_ops / (1e9 * (time.perf_counter() - st))
            t.set_description(f"*** loss: {loss:5.10f}   lr: {current_lr:.6f}"
                            f"   tm: {(et:=(time.perf_counter()-st))*1000:6.2f} ms {GlobalCounters.global_ops/(1e9*et):7.0f} GFLOPS")
            train_loss += loss

        gmt = time.perf_counter()
        GlobalCounters.reset()
        val_loss = val_step().float().item()
        get = time.perf_counter()
        current_lr = lr_sched.get_lr().item()

        print(f"\033[F*** epoch {epoch:3d}  GFLOPS: {current_gflops:7.0f}  tm: {(gmt-gst):5.2f} s    val_tm: {(get-gmt):5.2f} s lr: {current_lr:.6f}  train_loss: {train_loss/num_steps_per_epoch:5.10f}   val_loss: {val_loss:5.10f} @ {get-start_tm:6.2f} s  ")

    test_loss = test_step().float().item()
    print(f"*** test_loss: {test_loss:5.10f} @ {time.perf_counter()-start_tm:6.2f} s  ")