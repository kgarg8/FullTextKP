def compute_F1(prec, rec):
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

def metric_fn(metrics, args):
    composed_metric = {}
    total_data      = sum([metric['total_data'] for metric in metrics])

    for beam in [True, False]:
        if beam:
            beam_tag = "_beam"
        else:
            beam_tag = ""

        if "total_present_precision_beam" not in metrics[0] and beam:
            continue

        for topk in metrics[0]['total_present_precision']:
            total_present_precision = sum([metric["total_present_precision"+beam_tag][topk] for metric in metrics])
            total_present_recall    = sum([metric["total_present_recall"+beam_tag][topk] for metric in metrics])
            avg_present_precision   = total_present_precision / total_data
            avg_present_recall      = total_present_recall / total_data
            macro_present_F1        = compute_F1(avg_present_precision, avg_present_recall)
            
            composed_metric["present_precision_" + topk+beam_tag] = avg_present_precision
            composed_metric["present_recall_" + topk+beam_tag]    = avg_present_recall
            composed_metric["macro_present_F1_" + topk+beam_tag]  = macro_present_F1
            
            total_absent_precision = sum([metric["total_absent_precision"+beam_tag][topk] for metric in metrics])
            total_absent_recall    = sum([metric["total_absent_recall"+beam_tag][topk] for metric in metrics])
            avg_absent_precision   = total_absent_precision / total_data
            avg_absent_recall      = total_absent_recall / total_data
            macro_absent_F1        = compute_F1(avg_absent_precision, avg_absent_recall)
            
            composed_metric["absent_precision_" + topk+beam_tag] = avg_absent_precision
            composed_metric["absent_recall_" + topk+beam_tag]    = avg_absent_recall
            composed_metric["macro_absent_F1_" + topk+beam_tag]  = macro_absent_F1

    loss = sum([metric['loss'] * metric['total_data'] for metric in metrics]) / total_data
    composed_metric['loss'] = loss
    return composed_metric

def compose_dev_metric(metrics, args, config):
    total_metric = 0
    n = len(metrics)
    for key in metrics:
        total_metric += metrics[key][config["save_by"]]
    return config["metric_direction"] * total_metric / n    
