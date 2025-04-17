# DrugGPT/__init__.py
import os
import logging
import yaml
# Global imports
from DrugGPT.src.ensemble.ensemble_model import EnsembleModel
from DrugGPT.src.prompt.prompt_manager import PromptManager
from DrugGPT.src.llama.llama_utils import LLaMAUtils, SoftEmbedding
from DrugGPT.src.gcn.gcn_model import GraphConvolutionalNetwork
from DrugGPT.src.prompt_tuning.soft_prompt_tuning import SoftPromptTuner, extract_entities
from DrugGPT.src.gcn.dsdg import DSDGGenerator
from DrugGPT.src.utils import parser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DrugGPT')

# Load global configurations
config_path = 'configs/model.yaml'
with open(config_path, 'r') as file:
    global_config = yaml.safe_load(file)

# Set API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Initialize environment for OpenAI and Hugging Face APIs
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
else:
    logger.warning("OpenAI API key not found in environment variables.")

if HUGGINGFACEHUB_API_TOKEN:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
else:
    logger.warning("Hugging Face API token not found in environment variables.")

# Any other initialization code can go here

# Logging the successful initialization
logger.info("Initialized DrugGPT package successfully.")
