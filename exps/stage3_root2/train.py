import argparse
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
from tensorboardX import SummaryWriter

from cvpack.torch_modeling.engine.engine import Engine
from cvpack.utils.pyt_utils import ensure_dir

from config import cfg
from model.MMDA import MMDA
from lib.utils.dataloader import get_train_loader
from lib.utils.solver import make_lr_scheduler, make_optimizer


def main():
    parser = argparse.ArgumentParser()

    with Engine(cfg, custom_parser=parser) as engine:
        logger = engine.setup_log(
            name='train', log_dir=cfg.OUTPUT_DIR, file_name='log.txt')
        args = engine.args
        ensure_dir(cfg.OUTPUT_DIR)

        model = MMDA(cfg, run_efficient=cfg.RUN_EFFICIENT)
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        num_gpu = len(engine.devices) 
        #  default num_gpu: 8, adjust iter settings
        cfg.SOLVER.CHECKPOINT_PERIOD = \
                int(cfg.SOLVER.CHECKPOINT_PERIOD * 3 / num_gpu)
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * 3 / num_gpu)

        optimizer = make_optimizer(cfg, model, num_gpu)
        scheduler = make_lr_scheduler(cfg, optimizer)

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        if engine.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                broadcast_buffers=True, find_unused_parameters=True)

        if engine.continue_state_object:
            engine.restore_checkpoint(is_restore=False)
        else:
            if cfg.MODEL.WEIGHT:
                engine.load_checkpoint(cfg.MODEL.WEIGHT, is_restore=False)

        data_loader = get_train_loader(cfg, num_gpu=num_gpu, is_dist=engine.distributed,
                                       use_augmentation=True, with_mds=cfg.WITH_MDS)

        # -------------------- do training -------------------- #
        logger.info("\n\nStart training with pytorch version {}".format(
            torch.__version__))

        max_iter = len(data_loader)
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        if engine.local_rank == 0:
            tb_writer = SummaryWriter(cfg.TENSORBOARD_DIR)

        model.train()

        time1 = time.time()
        loss_total = 0
        loss_2d = 0
        loss_3d = 0
        loss_hm = 0
        m = 0
        for iteration, (images, valids, labels, xz_labels, yz_labels) in enumerate(
                data_loader, engine.state.iteration): 
            iteration = iteration + 1
            images = images.to(device)
            valids = valids.to(device)
            labels = labels.to(device)
            xz_labels = xz_labels.to(device)
            yz_labels = yz_labels.to(device)

            
            loss_dict = model(images, valids, labels, xz_labels, yz_labels)
            losses = loss_dict['total_loss']
            loss_total += loss_dict['total_loss'].item()
            loss_hm += loss_dict['loss_hm'].item()
            loss_3d += loss_dict['loss_3d'].item()
            loss_2d += loss_dict['loss_2d'].item()
            loss_tmp = dict(total_loss=loss_total, loss_2d=loss_2d, loss_3d=loss_3d, loss_hm=loss_hm)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

            del images, valids, labels, losses

            if engine.local_rank == 0:
                if iteration % 100 == 0 or iteration == max_iter:
                    log_str = 'Iter:%d, LR:%.1e, ' % (
                        iteration, optimizer.param_groups[0]["lr"] / num_gpu)
                    for key in loss_dict:
                        tb_writer.add_scalar(
                            key, loss_dict[key].mean(), global_step=iteration)
                        log_str += key + ': %.3f, ' % float(loss_tmp[key]/100)
                    
                    loss_total = 0
                    loss_2d = 0
                    loss_3d = 0
                    loss_hm = 0
                    time2 = time.time()
                    elapsed_time = time2 - time1
                    time1 = time2
                    required_time = elapsed_time / 100 * (max_iter - iteration)
                    hours = required_time // 3600
                    mins = required_time % 3600 // 60
                    log_str += 'To Finish: %dh%dmin,' % (hours, mins) 

                    logger.info(log_str)

            if iteration % checkpoint_period == 0 or iteration == max_iter:
                engine.update_iteration(iteration)
                if not engine.distributed or engine.local_rank == 0:
                    engine.save_and_link_checkpoint(cfg.OUTPUT_DIR)

            if iteration >= max_iter:
                logger.info('Finish training process!')
                break


if __name__ == "__main__":
    main()
