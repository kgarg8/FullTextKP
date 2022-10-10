import argparse, time, shutil
from extractor import PacSumExtractorWithBert, PacSumExtractorWithTfIdf
from data_iterator import Dataset
from utils import jsonl_save
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rep'             , type=str, choices = ['tfidf', 'bert']                         , help='tfidf or bert'          )
    parser.add_argument('--bert_model_file' , type=str, default='pacssum_models/pytorch_model_finetuned.bin', help='bert model file'        )
    parser.add_argument('--bert_config_file', type=str, default='pacssum_models/bert_config.json'           , help='bert configuration file')
    parser.add_argument('--bert_vocab_file' , type=str, default='pacssum_models/vocab.txt'                  , help='bert vocabulary file'   )

    parser.add_argument('--beta'   , type=float, default=0., help='beta'   )
    parser.add_argument('--lambda1', type=float, default=0., help='lambda1')
    parser.add_argument('--lambda2', type=float, default=1., help='lambda2')

    parser.add_argument('--train_load_file', type=str, default='../stage2_processed_data/v1/ACM_train.jsonl'   , help='data for training'  )
    parser.add_argument('--dev_load_file'  , type=str, default='../stage2_processed_data/v1/ACM_dev_ACM.jsonl' , help='data for validation')
    parser.add_argument('--test_load_file' , type=str, default='../stage2_processed_data/v1/ACM_test_ACM.jsonl', help='data for testing'   )

    parser.add_argument('--train_dump_file', type=str, default='ACM_train.jsonl'   )
    parser.add_argument('--dev_dump_file'  , type=str, default='ACM_dev_ACM.jsonl' )
    parser.add_argument('--test_dump_file' , type=str, default='ACM_test_ACM.jsonl')

    parser.add_argument('--dump_folder'    , type=str, default='processed_data/v1/')

    args = parser.parse_args()
    print(args)

    tic = time.perf_counter()
    if args.rep == 'tfidf':
        extractor = PacSumExtractorWithTfIdf(beta = args.beta, lambda1=args.lambda1, lambda2=args.lambda2)
        
        train_dataset          = Dataset(args.train_load_file)
        train_dataset_iterator = train_dataset.iterate_tfidf()
        train_objs             = extractor.extract_summary(train_dataset_iterator)
        
        dev_dataset            = Dataset(args.dev_load_file)
        dev_dataset_iterator   = dev_dataset.iterate_tfidf()
        dev_objs               = extractor.extract_summary(dev_dataset_iterator)
        
        test_dataset           = Dataset(args.test_load_file)
        test_dataset_iterator  = test_dataset.iterate_tfidf()
        test_objs              = extractor.extract_summary(test_dataset_iterator)

    elif args.rep == 'bert':
        extractor = PacSumExtractorWithBert(bert_model_file = args.bert_model_file, bert_config_file = args.bert_config_file, beta = args.beta, lambda1=args.lambda1, lambda2=args.lambda2)
        
        train_dataset          = Dataset(args.train_load_file, vocab_file = args.bert_vocab_file)
        train_dataset_iterator = train_dataset.iterate_bert()
        train_objs             = extractor.extract_summary(train_dataset_iterator)
        
        dev_dataset            = Dataset(args.dev_load_file, vocab_file = args.bert_vocab_file)
        dev_dataset_iterator   = dev_dataset.iterate_bert()
        dev_objs               = extractor.extract_summary(dev_dataset_iterator)
        
        test_dataset           = Dataset(args.test_load_file, vocab_file = args.bert_vocab_file)
        test_dataset_iterator  = test_dataset.iterate_bert()
        test_objs              = extractor.extract_summary(test_dataset_iterator)
        
    toc = time.perf_counter()
    print(f'Summarization finished in {toc - tic:0.4f} seconds')

    tic = time.perf_counter()
    Path(args.dump_folder).mkdir(parents=True, exist_ok=True)
    
    jsonl_save(filepath=args.dump_folder+args.train_dump_file, data_dict=train_objs)
    jsonl_save(filepath=args.dump_folder+args.dev_dump_file  , data_dict=dev_objs  )
    jsonl_save(filepath=args.dump_folder+args.test_dump_file , data_dict=test_objs )
    
    toc = time.perf_counter()
    print(f'Dump finished in {toc - tic:0.4f} seconds')

    shutil.copy('../stage2_processed_data/v1/ACM_metadata.pkl', args.dump_folder)  # Copy metadata file as it is
