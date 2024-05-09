import logging
import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score


def train(model, criterion,CEcriterion, optimizer, scheduler, n_labels,
      train_loader, val_loader, device, writer, checkpoints_dir,
      accum_steps=1, print_every=1000, n_epoch=10,
      save_from_epoch=1, start_epoch=1, best_acc=0., step_type="step1",mlm_param=1,

    ):
    step = 0
    logging.info("Start training...")
    for epoch in range(start_epoch, n_epoch + 1):
        model.train()
        epoch_loss_train = 0
        start = time.time()
        for batch_id, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, logits = run_batch(model,criterion, CEcriterion,batch, device, step_type,mlm_param)
            loss.backward()
            if batch_id % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                writer.add_scalar('lr', np.array(scheduler.get_last_lr()), step)

            epoch_loss_train += loss.item()

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = batch['label'].numpy()
            current_acc = accuracy_score(labels, preds)

            if batch_id % print_every == 0 and batch_id > 0:
                logging.info('EPOCH {} BATCH {} of {}\t TRAIN_LOSS {:.3f}'.format(
                    epoch, batch_id, len(train_loader),epoch_loss_train / batch_id)
                )
                logging.info(
                    f'EPOCH TIME: {(time.time() - start) // 60} min {round((time.time() - start) % 60, 1)} sec')
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/acc', current_acc, step)
            step += 1
        logging.info(
            'EPOCH {} TRAIN_LOSS {:.3f}'.format(epoch, epoch_loss_train / len(train_loader)))
        logging.info(f'EPOCH TIME: {(time.time() - start) // 60} min {round((time.time() - start) % 60, 1)} sec')

        y_val, pred_val, _, loss_val = validate(model, criterion,val_loader, n_labels, device)
        epoch_accuracy = accuracy_score(y_val, pred_val)
        logging.info('-' * 100)
        logging.info('EVAL EPOCH {}\t EVAL_LOSS {:.3f}\tACCURACY {:.3f}'.format(epoch, loss_val / len(val_loader),
                                                                                epoch_accuracy))
        logging.info(f'EVAL EPOCH TIME: {(time.time() - start) // 60} min {round((time.time() - start) % 60, 1)} sec')

        if epoch >= save_from_epoch:
            torch.save({'model': model, 'loss_val': loss_val, 'acc_val': epoch_accuracy, 'epoch': epoch},
                       os.path.join(checkpoints_dir, f'checkpoint_e{epoch}.pt'))
        if epoch_accuracy >= best_acc:
            torch.save({'model': model, 'loss_val': loss_val, 'acc_val': epoch_accuracy, 'epoch': epoch},
                       os.path.join(checkpoints_dir, 'checkpoint_best_acc.pt'))
            best_acc=epoch_accuracy
        logging.info(f'model saved')


def validate(model, criterion,loader, n_labels, device):
    model.eval()

    loss_val = 0
    pred_val = np.zeros(len(loader.dataset))
    y_val = np.zeros(len(loader.dataset))
    logits_val = np.zeros((len(loader.dataset), n_labels))
    with torch.no_grad():
        batch_size = loader.batch_size
        for i, batch in enumerate(loader):
            y_batch = batch['label']
            loss, logits = test_batch(model,criterion, batch, device)
            loss_val += loss.item()
            upper_bound = min((i + 1) * batch_size, len(loader.dataset))
            logits_val[i * batch_size:upper_bound, :] = logits.detach().cpu().numpy()
            pred_intent = np.argmax(logits.detach().cpu().numpy(), axis=1)
            pred_val[i * batch_size:upper_bound] = pred_intent
            y_val[i * batch_size:upper_bound] = y_batch
        return y_val, pred_val, logits_val, loss_val

def test_batch(model,criterion, batch, device):

    logits= model(batch['pair_ids'].to(device), batch['pair_attention_mask'].to(device))
    loss = criterion(logits, batch['label_enc'].to(device))

    return loss, logits


def run_batch(model,criterion,CEcriterion, batch, device, step_type,mlm_param):
    if step_type=="step1":
        # mlm_loss
        mlm_loss = model.mlmForward(batch['mask_input_ids'].to(device),batch['attention_mask'].to(device),batch['mask_lb'].to(device))
        # model_loss
        logits= model(batch['pair_ids'].to(device), batch['pair_attention_mask'].to(device))
        cls_loss = criterion(logits, batch['label_enc'].to(device))
        loss = cls_loss + mlm_param*mlm_loss
    elif step_type=="step2":
        logits= model(batch['pair_ids'].to(device), batch['pair_attention_mask'].to(device))
        cls_loss = criterion(logits, batch['label_enc'].to(device))
        loss=cls_loss

    return loss, logits




