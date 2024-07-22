import time

import numpy as np
import torch

import wandb
from evaluation import evaluation
from model import ProtoClassifier, ResModel
from src.feature_memory import *
from src.contrastive_losses import contrastive_target_source, contrastive_class_to_class_learned_memory, contrastive_source_target
from util import (
    TIMING_TABLE,
    BaseTrainerConfig,
    LR_Scheduler,
    MetricMeter,
    SLATrainerConfig,
)


class BaseDATrainer:
    def __init__(self, loaders, args, backbone="resnet34"):
        self.model = ResModel(backbone, output_dim=args.dataset["num_classes"]).cuda()
        self.params = self.model.get_params(args.lr)
        self.optimizer = torch.optim.SGD(
            self.params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
        self.lr_scheduler = LR_Scheduler(self.optimizer, args.num_iters)

        # `self.iter_loaders` is used to load the training data. However, during evaluation or testing,
        # we need to pass a specific data loader that is not available in an iterator.
        self.loaders = loaders
        self.iter_loaders = iter(loaders)

        # recording
        self.meter = MetricMeter()

        # required arguments for DATrainer
        self.config = BaseTrainerConfig.from_args(args)
        # officehome 64 domainnet512
        self.max_features_per_class = 64 if args.dataset == 'OfficeHome' else 512
        self.feature_memory = FeatureLabelMemoryBank(dataset=args.dataset, num_classes=args.dataset["num_classes"], max_features_per_class=self.max_features_per_class, feature_dim=512)

    def get_source_loss(self, step, *data):
        return self.model.base_loss(*data)

    def get_target_loss(self, step, *data):
        return self.model.base_loss(*data)

    def logging(self, step, info, unit="min"):
        wandb.log(
            {
                **info,
                "iteration": step,
                f"running time ({unit})": (time.perf_counter() - self.meter.start_time)
                * TIMING_TABLE[unit],
            }
        )

    def evaluate(self):
        val_acc = evaluation(self.loaders.loaders["target_validation"], self.model)
        t_acc = evaluation(self.loaders.loaders["target_unlabeled_test"], self.model)
        if val_acc >= self.meter.best_val_acc:
            self.meter.best_val_acc = val_acc
            self.meter.counter = 0
            self.meter.best_acc = t_acc
        else:
            self.meter.counter += 1
        return val_acc, t_acc

    def training_step(self, step, *data):
        sx, sy, tx, ty, _ = data
        self.optimizer.zero_grad()
        s_loss = self.get_source_loss(step, sx, sy)
        t_loss = self.get_target_loss(step, tx, ty)

        loss = (s_loss + t_loss) / 2
        loss.backward()
        self.optimizer.step()

        return s_loss.item(), t_loss.item(), 0

    def train(self):
        self.model.train()

        self.meter.start_time = time.perf_counter()
        for step in range(1, self.config.num_iters + 1):
            (sx, sy), (tx, ty), (ux, _) = next(self.iter_loaders)
            s_loss, t_loss, u_loss = self.training_step(step, sx, sy, tx, ty, ux)
            self.lr_scheduler.step()

            # logging
            if step % self.config.log_interval == 0:
                self.logging(
                    step,
                    {
                        "LR": self.lr_scheduler.get_lr(),
                        "source loss": s_loss,
                        "target loss": t_loss,
                        "unlabeled loss": u_loss,
                    },
                )

            # early-stopping & evaluation
            if step >= self.config.early and step % self.config.eval_interval == 0:
                eval_acc, t_acc = self.evaluate()
                self.logging(
                    step,
                    {
                        "evaluation accuracy": eval_acc,
                        "test accuracy": t_acc,
                    },
                )
                wandb.run.summary["best_test_accuracy"] = self.meter.best_acc

            # early-stopping
            # Here we set a huge number to plot the whole testing procedure.
            # Change it to a reasonable value for early-stopping
            if self.meter.counter > 10000 or step == self.config.num_iters:
                break


class UnlabeledDATrainer(BaseDATrainer):
    def __init__(self, loaders, args, backbone="resnet34", unlabeled_method="mme"):
        super().__init__(loaders, args, backbone)
        self.unlabeled_method = unlabeled_method

    def unlabeled_training_step(self, step, ux):
        self.optimizer.zero_grad()
        unlabeled_loss_fn = getattr(self.model, f"{self.unlabeled_method}_loss")
        u_loss = unlabeled_loss_fn(step, self.feature_memory, *ux)
        u_loss.backward()
        self.optimizer.step()

        return u_loss.item()

    def training_step(self, step, sx, sy, tx, ty, ux):
        s_loss, t_loss, _ = super().training_step(step, sx, sy, tx, ty, ux)
        u_loss = self.unlabeled_training_step(step, ux)
        return s_loss, t_loss, u_loss


def get_SLA_trainer(base_class):
    class SLADATrainer(base_class):
        def __init__(self, loaders, args, **kwargs):
            super().__init__(loaders, args, **kwargs)
            self.config = SLATrainerConfig.from_args(args)
            self.ppc = ProtoClassifier(args.dataset["num_classes"])
            self.num_classes = args.dataset["num_classes"]



        def get_source_loss(self, step, sx, sy):
            sf = self.model.get_features(sx)
            contra_source_loss = 0
            if step > self.config.warmup:
                contra_source_loss = contrastive_class_to_class_learned_memory(sf, sy, self.feature_memory)
            if step > self.config.warmup - 2000:
                with torch.no_grad():
                    sim = F.normalize(sf) @ self.feature_memory.proto_s.mo_pro.T
                    _, pl_pro = torch.max(sim, dim=1)
                    out = self.model.get_predictions(sf)
                    prob = F.softmax(out, dim=1)
                    mp, pr = torch.max(prob, dim=1)
                    mask = (mp.ge(0.95).float()) * ((pr == sy).float()) * ((pl_pro == sy).float())
                    for feature, label in zip(sf[mask.bool()], sy[mask.bool()]):
                        self.feature_memory.insert_feature_label(feature, label)
            if step > self.config.warmup:
                sy2 = self.ppc(sf.detach(), self.config.T)
                s_loss = self.model.sla_loss(sf, sy, sy2, self.config.alpha)
            else:
                s_loss = self.model.feature_base_loss(sf, sy)

            return s_loss + contra_source_loss

        def get_target_loss(self, step, tx, ty):
            tf = self.model.get_features(tx)
            t_loss = self.model.feature_base_loss(tf, ty)
            return t_loss

        def ppc_update(self, step):
            if step == self.config.warmup:
                self.ppc.init(self.model, self.loaders.loaders["target_unlabeled_test"])
                self.lr_scheduler.refresh()

            # if step > self.config.warmup and step % self.config.update_interval == 0:
            if step != self.config.warmup and step % self.config.update_interval == 0:
                self.ppc.init(self.model, self.loaders.loaders["target_unlabeled_test"])

        def training_step(self, step, *data):
            s_loss, t_loss, u_loss = super().training_step(step, *data)
            self.ppc_update(step)

            return s_loss, t_loss, u_loss

    return SLADATrainer


def get_trainer(base_class, label_trick=None):
    match label_trick:
        case "SLA", *_:
            return get_SLA_trainer(base_class)
        case _:
            return base_class
