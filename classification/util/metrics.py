from sklearn.metrics import accuracy_score, f1_score


def build_main_report(targets, predictions, labels=None, prefix=''):
    return {
        f'{prefix}accuracy': accuracy_score(targets, predictions),
        f'{prefix}f1_weighted': f1_score(targets, predictions, labels=labels, average='weighted'),
        f'{prefix}f1_macro': f1_score(targets, predictions, labels=labels, average='macro')
    }
