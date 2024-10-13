import hydra
from hydra.utils import to_absolute_path
import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel,AutoModelForMaskedLM
import sys
sys.path.append("../../")
print(sys.path)
from gzsl.classification.data_utils.dataset import get_dataset
from gzsl.classification.model.losses import get_loss,get_ce_loss
from gzsl.classification.model.model import IntentClassifierNLI,SoftpromptIntentClassifierNLI
from gzsl.classification.util.environment import seed_everything
from gzsl.classification.util.loops import validate
from gzsl.classification.util.preprocessing import read_intent_info, read_decomposed_intents, read_split_data
from gzsl.classification.util.metrics import build_main_report



@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    logging.basicConfig(level=logging.INFO)
    seed_everything(cfg.experiment.seed)
    intent_info_path = Path(to_absolute_path(cfg.dataset.intent_info_path))
    intents, descriptions = read_intent_info(intent_info_path, cfg.dataset.description_type,cfg.dataset.name)
    concepts, actions = read_decomposed_intents(intent_info_path)

    logging.info(f'Read and preprocesed data from {cfg.dataset.path}')
    data_path = Path(to_absolute_path(cfg.dataset.path))
    zeroshot_intents, train_df, dev_df, test_df = read_split_data(data_path, intents, cfg.experiment.seed)
    logging.info(f'Unseen intents: {zeroshot_intents}')
    seen_intents = [intent for intent in intents if intent not in zeroshot_intents]

    report_dir = cfg.experiment.test_report_dir or './'
    os.makedirs(report_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(to_absolute_path(cfg.model.base_model))
    test_loader = DataLoader(
        get_dataset(cfg, test_df, tokenizer, intents, descriptions, concepts, actions,"evaluate",cfg.experiment.mlm_percent),
        batch_size=cfg.experiment.batch_size
    )



    device = torch.device('cuda:'+str(cfg.experiment.cuda) if torch.cuda.is_available() else 'cpu')

    logging.info("-" * 50)
    logging.info(f'seed: {cfg.experiment.seed}')
    logging.info(f'device: {device}')
    logging.info(f'description_type: {cfg.dataset.description_type}')
    logging.info(f'embedding_params: {cfg.experiment.embedding_param}')
    logging.info(f'k_negs: {cfg.experiment.k_negative}')
    logging.info(f'mlm_percents: {cfg.experiment.mlm_percent}')
    logging.info("-" * 50)



    base_model = AutoModelForMaskedLM.from_pretrained(to_absolute_path(cfg.model.base_model)).to(device)
    if hasattr(base_model, 'model'):
        base_model = base_model.model

    if cfg.experiment.step=="step1":
        model = IntentClassifierNLI(base_model, hidden_size=cfg.model.embedding_dim, dropout=cfg.model.dropout).to(
        device)


    elif cfg.experiment.step=="step2":
        IntentClassifier = IntentClassifierNLI(base_model, hidden_size=cfg.model.embedding_dim, dropout=cfg.model.dropout).to(
            device)
        model = SoftpromptIntentClassifierNLI(IntentClassifier, hidden_size=cfg.model.embedding_dim,
                                                  dropout=cfg.model.dropout,softprompt_length=cfg.experiment.prompt_len, device=device).to(device)

    model_name = f'checkpoint_e{cfg.experiment.test_epoch}' if cfg.experiment.test_epoch else 'checkpoint_best_acc'
    model_path = os.path.join(os.getcwd(), 'checkpoints', model_name + '.pt')
    logging.info(f'Loading model: {model_path}')
    model.load_state_dict(torch.load(model_path)['model'].state_dict())

    criterion = get_loss(cfg)


    logging.info('Start evaluation...')
    y_test, pred_test, logits_test, _ = validate(model, criterion, test_loader, len(intents), device)
    preds_test = np.array([intents[int(p)] for p in pred_test])
    true_labels_test = np.array([intents[int(l)] for l in y_test])
    logging.info(classification_report(true_labels_test, preds_test))
    indexes_seen = [i for i, intent in enumerate(true_labels_test) if intent not in zeroshot_intents]
    indexes_unseen = [i for i, intent in enumerate(true_labels_test) if intent in zeroshot_intents]

    report = build_main_report(true_labels_test, preds_test)
    report_unseen = build_main_report(true_labels_test[indexes_unseen], preds_test[indexes_unseen],
                                      labels=zeroshot_intents, prefix="unseen_")
    report.update(report_unseen)

    report_seen = build_main_report(true_labels_test[indexes_seen], preds_test[indexes_seen],
                                    labels=seen_intents, prefix="seen_")
    report.update(report_seen)

    report['classification_report'] = classification_report(true_labels_test, preds_test, output_dict=True)

    with open(f'{report_dir}/report_{model_name}.json', 'w') as f:
        json.dump(report, f)

    with open(f'{report_dir}/preds_{model_name}.json', 'w') as f:
        json.dump({
            'true': true_labels_test.tolist(),
            'pred': preds_test.tolist(),
            'intents': intents,
            'zeroshot_intents': zeroshot_intents
        }, f)

    np.save(f'{report_dir}/logits_{model_name}.npy', logits_test)

    logging.info(f'Accuracy: {report["accuracy"]}')
    logging.info(f'Accuracy unseen: {report["unseen_accuracy"]}')
    logging.info(f'Accuracy seen: {report["seen_accuracy"]}')
    logging.info(f'F1 weighted: {report["f1_weighted"]}')
    logging.info(f'F1 macro unseen: {report["unseen_f1_macro"]}')


if __name__ == '__main__':
    main()
