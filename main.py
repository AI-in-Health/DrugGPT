import pandas as pd
import os
import logging
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
from src.ensemble.ensemble_model import EnsembleModel
from src.prompt.prompt_manager import PromptManager
from src.llama.llama_utils import LLaMAUtils
from src.gcn.dsdg import DSDGGenerator
from src.gcn.gcn_model import GraphConvolutionalNetwork
from src.utils.parser import binary_parser, mc_parser, text_parser, double_binary_parser
from src.utils.prompt_formatter import FORMATTERS
from src.evaluation.evaluation_metrics import Evaluation
from src.utils.preprocessing import preprocess_query
from dotenv import load_dotenv




load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
PC_INDEX = os.getenv("PC_INDEX")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


DATASET_MAPPING = {
    'ADE': {'type': 'text', 'parser': text_parser},
    'ChatDoctor': {'type': 'text', 'parser': text_parser},
    'COVID': {'type': 'binary', 'parser': binary_parser},
    'DDI': {'type': 'binary', 'parser': binary_parser},
    'Drug_Effects': {'type': 'double_binary', 'parser': double_binary_parser},
    'DrugBank_Class': {'type': 'mc', 'parser': mc_parser},
    'MedMCQA': {'type': 'mc', 'parser': mc_parser},
    'Mimic': {'type': 'mc', 'parser': mc_parser},
    'MMLU': {'type': 'mc', 'parser': mc_parser},
    'PubMedQA': {'type': 'binary', 'parser': binary_parser},
    'USMLE': {'type': 'mc', 'parser': mc_parser}
}


def load_dataset(dataset_name, data_path='data/'):
    """
    Load a dataset from the data directory.
    
    Args:
        dataset_name (str): Name of the dataset to load
        data_path (str): Path to the data directory
        
    Returns:
        dict: Dictionary containing dataset samples, labels, and other relevant fields
    """
    file_path = os.path.join(data_path, f"{dataset_name}.csv")
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {dataset_name} dataset with {len(df)} samples")
        
        
        result = {
            'sample': df['question'].tolist(),
            'label': df['answer'].tolist()
        }
        
        
        if 'context' in df.columns:
            result['context'] = df['context'].tolist()
            
        
        for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
            if opt in df.columns:
                result[opt] = df[opt].tolist()
            
        return result
                
    except Exception as e:
        logging.error(f"Error loading {dataset_name} dataset: {e}")
        return None


def prepare_formatters_and_parsers():
    """
    Prepare formatters and parsers for all dataset types.
    
    Returns:
        tuple: (formatters_dict, parsers_dict) dictionaries with formatters and parsers
    """
    
    formatters_dict = FORMATTERS
    
    
    parsers_dict = {dataset: info['parser'] for dataset, info in DATASET_MAPPING.items()}
    
    return formatters_dict, parsers_dict


def evaluate_dataset(dataset_name, dataset, ensemble_model, parser, formatter, cot_enabled=True, use_openai=True, use_db=True, sample_size=50):
    """
    Evaluate a dataset using the ensemble model.
    
    Args:
        dataset_name (str): Name of the dataset
        dataset (dict): Dataset with 'sample' and 'label' keys
        ensemble_model (EnsembleModel): The ensemble model
        parser (function): Parser function for this dataset type
        formatter (function): Formatter function for this dataset type
        cot_enabled (bool): Whether Chain of Thought is enabled
        use_openai (bool): Whether to use OpenAI for inference
        use_db (bool): Whether to use database for knowledge retrieval
        sample_size (int): Number of samples to evaluate from each dataset
        
    Returns:
        dict: Results of the evaluation
    """
    logging.info(f"Evaluating {dataset_name} dataset with CoT {'enabled' if cot_enabled else 'disabled'}")
    
    slice_size = min(len(dataset['sample']), sample_size)
    accurate_predictions = 0
    precision_list = []
    recall_list = []
    f1_list = []
    
    
    results_data = {
        'Question': [],
        'Prediction': [],
        'Actual': [],
        'Correct': []
    }
    
    has_context = 'context' in dataset and len(dataset['context']) > 0
    
    for i in tqdm(range(slice_size), desc=f"Processing {dataset_name}"):
        try:
            
            input_data = dataset['sample'][i]
            
            
            row = {'question': input_data}
            
            
            if has_context and i < len(dataset['context']):
                row['context'] = dataset['context'][i]
            
            
            if dataset_name in ['Mimic', 'MedMCQA', 'DrugBank_Class', 'USMLE', 'MMLU']:
                for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
                    if opt in dataset and len(dataset[opt]) > i:
                        row[opt] = dataset[opt][i]
            
            
            processed_input = preprocess_query(row)
            
            
            formatted_input = formatter(processed_input)
            
            
            full_response = ensemble_model.run_inference(
                input_data=formatted_input,
                use_openai=use_openai,
                use_db=use_db
            )
            
            
            analysis, prediction = parser(full_response)
            
            
            actual_label = dataset['label'][i].lower()
            
            if dataset_name == 'ChatDoctor':
                
                precision, recall, f1 = Evaluation.calculate_f1_metrics(prediction, actual_label)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                correct = f1 > 0.2 
            elif dataset_name == 'Drug_Effects':
                
                prediction_parts = prediction.split(', ')
                actual_parts = actual_label.split(', ')
                correct = prediction_parts[0] == actual_parts[0] and prediction_parts[1] == actual_parts[1]
            else:
                
                correct = Evaluation.check_text_accuracy(prediction, actual_label)
            
            if correct:
                accurate_predictions += 1
            
            
            results_data['Question'].append(input_data)
            results_data['Prediction'].append(prediction)
            results_data['Actual'].append(actual_label)
            results_data['Correct'].append(correct)
            
            
            if i % 20 == 0:
                logging.info(f"Example {i}/{slice_size}: Correct={correct}")
                
        except Exception as e:
            logging.error(f"Error processing sample {i}: {e}")
            
            results_data['Question'].append(input_data if 'input_data' in locals() else "Error")
            results_data['Prediction'].append(f"ERROR: {str(e)}")
            results_data['Actual'].append(dataset['label'][i])
            results_data['Correct'].append(False)
    
    
    accuracy = accurate_predictions / slice_size
    
    
    results = {
        'Accuracy': accuracy,
        'Dataset': dataset_name,
        'CoT': cot_enabled,
        'Samples': slice_size
    }
    
    
    if dataset_name == 'ChatDoctor' and precision_list:
        results['Average Precision'] = sum(precision_list) / len(precision_list)
        results['Average Recall'] = sum(recall_list) / len(recall_list)
        results['Average F1 Score'] = sum(f1_list) / len(f1_list)
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cot_status = "cot" if cot_enabled else "no_cot"
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f"results/{dataset_name}_{cot_status}_{timestamp}.csv", index=False)
    
    logging.info(f"Results for {dataset_name} with CoT {'enabled' if cot_enabled else 'disabled'}: {results}")
    return results


def main():
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    
    parser = argparse.ArgumentParser(description="Evaluate DrugGPT across multiple datasets")
    parser.add_argument('--openai_key', type=str, help='OpenAI API key')
    parser.add_argument('--hf_key', type=str, help='Hugging Face API key')
    parser.add_argument('--excel_path', type=str, help='Path to DSDG Excel file')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_MAPPING.keys()),
                        help='List of datasets to evaluate (default: all)')
    parser.add_argument('--no_cot', action='store_true', help='Disable Chain of Thought')
    parser.add_argument('--use_openai', action='store_true', help='Use OpenAI API for inference')
    parser.add_argument('--use_db', action='store_true', help='Use database for knowledge retrieval (Pinecone)')
    parser.add_argument('--sample_size', type=int, default=50, help='Number of samples to evaluate from each dataset')
    args = parser.parse_args()
    
    
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.hf_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.hf_key
    
    
    if "OPENAI_API_KEY" not in os.environ:
        logging.error("OPENAI_API_KEY not set. Please provide --openai_key or set the environment variable.")
        return
    
    
    try:
        with open('configs/model.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading model configuration: {e}")
        return
    
    LLAMA_CONFIGS = configs['LLAMA_CONFIGS']
    GCN_CONFIGS = configs['GCN_CONFIGS']
    
    
    components = initialize_components(args, LLAMA_CONFIGS, GCN_CONFIGS)
    if not components:
        return
    
    ensemble_model = components['ensemble_model']
    
    
    formatters_dict, parsers_dict = prepare_formatters_and_parsers()
    
    
    all_results = []
    
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_MAPPING:
            logging.warning(f"Unknown dataset: {dataset_name}. Skipping.")
            continue
        
        logging.info(f"Processing dataset: {dataset_name}")
        
        
        dataset = load_dataset(dataset_name)
        if not dataset:
            logging.error(f"Failed to load {dataset_name} dataset. Skipping.")
            continue
        
        
        dataset_type = DATASET_MAPPING[dataset_name]['type']
        parser = DATASET_MAPPING[dataset_name]['parser']
        
        
        if not args.no_cot:
            
            formatter = formatters_dict[dataset_type]['cot']
            cot_enabled = True
        else:
            
            formatter = formatters_dict[dataset_type]['no_cot']
            cot_enabled = False
        
        results = evaluate_dataset(
            dataset_name, 
            dataset, 
            ensemble_model, 
            parser, 
            formatter, 
            cot_enabled=cot_enabled, 
            use_openai=args.use_openai,
            use_db=args.use_db,
            sample_size=args.sample_size
        )
        all_results.append(results)
    
    
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"results/all_results_{timestamp}.csv", index=False)
    
    
    logging.info("\n=== EVALUATION SUMMARY ===")
    for result in all_results:
        cot_status = "CoT enabled" if result['CoT'] else "CoT disabled"
        logging.info(f"{result['Dataset']} ({cot_status}): Accuracy = {result['Accuracy']:.4f}")


def initialize_components(args, LLAMA_CONFIGS, GCN_CONFIGS):
    """
    Initialize all required components for the evaluation.
    
    Args:
        args: Command line arguments
        LLAMA_CONFIGS: LLaMA model configurations from YAML
        GCN_CONFIGS: GCN model configurations from YAML
        
    Returns:
        dict: Dictionary of initialized components or None on failure
    """
    try:
        
        prompt_manager = PromptManager()
        
        
        knowledge_base = None
        soft_prompt_generator = None
        llama_utils = None
        
        
        if not args.use_db and args.excel_path:
            try:
                knowledge_base = DSDGGenerator(args.excel_path, embd_model_name='all-MiniLM-L6-v2')
                logging.info(f"Initialized DSDG generator with {args.excel_path}")
                
                
                soft_prompt_generator = GraphConvolutionalNetwork(
                    GCN_CONFIGS['input_dim'], 
                    GCN_CONFIGS['hidden_dim'],
                    GCN_CONFIGS['output_dim']
                )
                logging.info("Initialized soft prompt generator")
            except Exception as e:
                logging.warning(f"Could not initialize knowledge base: {str(e)}")
        
        
        if not args.use_openai:
            try:
                llama_utils = LLaMAUtils(LLAMA_CONFIGS)
                logging.info("Initialized LLaMA utils")
            except Exception as e:
                logging.warning(f"Could not initialize LLaMA utils: {str(e)}")
                args.use_openai = True  
        
        
        ensemble_model = EnsembleModel(
            prompt_manager=prompt_manager, 
            soft_prompt_generator=soft_prompt_generator, 
            knowledge_base=knowledge_base, 
            llama_utils=llama_utils, 
            openai_api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            pinecone_api_key=PINECONE_API_KEY,
            embedding_model=EMBEDDING_MODEL,
            pinecone_index=PC_INDEX
        )
        
        logging.info("Successfully initialized ensemble model and components")
        return {
            'prompt_manager': prompt_manager,
            'knowledge_base': knowledge_base,
            'soft_prompt_generator': soft_prompt_generator,
            'llama_utils': llama_utils,
            'ensemble_model': ensemble_model
        }
    except Exception as e:
        logging.error(f"Error initializing components: {e}")
        return None


if __name__ == "__main__":
    main()
