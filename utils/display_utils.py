import random

def step_display_fn(epoch, iter, item, args, config):
    display_string = "Model: {}, (LR: {}), Dataset: {}, Epoch: {}, Step: {}, Loss: {:.3f}".format(config['model_name'], 
                        config['current_lr'], args.dataset, epoch, iter, item['metrics']['loss'])
    return display_string

def example_display_fn(epoch, iter, item, args, config):
    item_len       = len(item['display_items']['predictions'])
    chosen_id      = random.choice([id for id in range(item_len)])
    display_string = "Example: \nSource: {}\nTarget: {}\nPrediction: {}\n".format(
        " ".join(item['display_items']['source'][chosen_id]),
        " ".join(item['display_items']['target'][chosen_id]),
        item['display_items']['predictions'][chosen_id])

    return display_string

def display(display_string, log_paths):
    with open(log_paths['log_path'], 'a') as fp:
        fp.write(display_string)
    with open(log_paths['verbose_log_path'], 'a') as fp:
        fp.write(display_string)
    print(display_string)