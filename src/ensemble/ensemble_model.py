from openai import OpenAI
import re
from pinecone import Pinecone
from ..prompt_tuning.soft_prompt_tuning import extract_entities
import logging


logging.getLogger("httpx").setLevel(logging.WARNING)

class EnsembleModel:
    """
    Orchestrates the process of inquiry analysis, knowledge acquisition, and evidence generation
    by integrating different models and knowledge sources.

    Attributes:
        prompt_manager (PromptManager): Manages and generates prompts for various tasks.
        soft_prompt_generator (GraphConvolutionalNetwork): GCN model to generate soft prompts dynamically.
        knowledge_base (DSDGGenerator): Knowledge base containing medical information.
        llama_utils (LLaMAUtils): Utility class for LLaMA model operations.
        openai_api_key (str): API key for OpenAI services.
        pinecone_api_key (str, optional): API key for Pinecone vector database.
        embedding_model (str, optional): OpenAI embedding model to use.
        pinecone_index (str, optional): Name of the Pinecone index to use.

    Methods:
        get_embedding(text): Generates embedding vector for the given text using OpenAI's embedding model.
        openai_inference(prompt): Conducts inference using OpenAI's GPT-3.5 model.
        llama_inference(prompt, identified_entities): Performs inference using LLaMA model with dynamic soft prompts.
        extract_knowledge(ka_response): Extracts relevant knowledge entries from the KA response.
        extract_knowledge_db(query_text): Retrieves relevant knowledge from Pinecone vector database.
        run_inference(input_data, use_openai, use_db): Orchestrates the complete inference process involving IA, KA, and EG steps.
    """

    def __init__(self, prompt_manager, soft_prompt_generator, knowledge_base, llama_utils, 
                 openai_api_key, model="gpt-3.5-turbo", pinecone_api_key=None, 
                 embedding_model=None, pinecone_index=None):
        self.prompt_manager = prompt_manager
        self.soft_prompt_generator = soft_prompt_generator
        self.knowledge_base = knowledge_base
        self.llama_utils = llama_utils
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=self.openai_api_key)
        self.model = model
        
        
        self.pinecone_api_key = pinecone_api_key
        self.embedding_model = embedding_model
        self.pinecone_index = pinecone_index
        self.pc = None
        if self.pinecone_api_key:
            self.pc = Pinecone(api_key=self.pinecone_api_key)

    def get_embedding(self, text):
        """Generate embedding vector for the given text using OpenAI's embedding model."""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def openai_inference(self, prompt, model=None):
        if model is None:
            model = self.model
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def llama_inference(self, prompt, identified_entities=None, use_soft_prompt=True):
        """
        Perform inference using the LLaMA model, optionally with soft prompts.
        
        Args:
            prompt (str): The input prompt
            identified_entities (list, optional): List of entities to use for soft prompt generation
            use_soft_prompt (bool): Whether to use soft prompts
            
        Returns:
            str: Model response
        """
        if use_soft_prompt and self.soft_prompt_generator and identified_entities:
            
            soft_prompt_prefix = self.soft_prompt_generator.generate_prefix(
                self.knowledge_base.get_graph(), identified_entities)
            
            
            updated_prompt = self._apply_soft_prompt(prompt, soft_prompt_prefix)
            return self.llama_utils.llama_inference(updated_prompt)
        else:
            
            return self.llama_utils.llama_inference(prompt)

    def _apply_soft_prompt(self, prompt, soft_prompt_prefix):
        """
        Apply a soft prompt prefix to the input prompt.
        
        Args:
            prompt (str): The original prompt
            soft_prompt_prefix (str): The soft prompt prefix to prepend
            
        Returns:
            str: The combined prompt
        """
        
        return soft_prompt_prefix + " " + prompt

    def extract_knowledge(self, ka_response):
        
        drug_knowledge_needed = re.findall(r'Drug Knowledge Needed \[([\d, ]+)\]', ka_response)
        disease_knowledge_needed = re.findall(r'Disease Knowledge Needed \[([\d, ]+)\]', ka_response)

        
        drug_numbers = [int(num) for num in drug_knowledge_needed[0].split(',')] if drug_knowledge_needed else []
        disease_numbers = [int(num) for num in
                           disease_knowledge_needed[0].split(',')] if disease_knowledge_needed else []

        
        knowledge_entries = []
        for num in drug_numbers:
            knowledge_entries.append(self.knowledge_base.dsdg_dict['drug'][f'Drug {num}'])
        for num in disease_numbers:
            knowledge_entries.append(self.knowledge_base.dsdg_dict['disease'][f'Disease {num}'])

        
        combined_knowledge = '\n'.join(knowledge_entries)
        return combined_knowledge

    def extract_knowledge_db(self, query_text, top_k=3):
        """
        Retrieve relevant knowledge from vector database using embedding similarity.
        
        Args:
            query_text (str): The text to search for in the database
            top_k (int): Number of top results to retrieve from each namespace
            
        Returns:
            str: Combined knowledge from database matches
        """
        if not self.pc or not self.pinecone_index:
            raise ValueError("Pinecone is not configured. Please provide API key and index name during initialization.")
        
        
        query_embedding = self.get_embedding(query_text)
        
        
        index = self.pc.Index(self.pinecone_index)
        results_conditions = index.query(
            namespace="conditions",
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        results_drugs = index.query(
            namespace="drugs",
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        
        knowledge = ""
        
        
        if results_conditions.matches:
            knowledge += f"CONDITION INFORMATION:\n"
            for i, match in enumerate(results_conditions.matches):
                condition_info = match.metadata
                knowledge += f"--- Condition Match {i+1} (Score: {match.score:.4f}) ---\n"
                knowledge += f"Condition: {condition_info.get('condition', 'N/A')}\n"
                knowledge += f"URL: {condition_info.get('url', 'N/A')}\n"
                knowledge += f"Text: {condition_info.get('text', 'N/A')}\n"
                if 'sections' in condition_info:
                    knowledge += f"Available sections: {', '.join(eval(condition_info['sections']))}\n"
                knowledge += "\n"
        
        
        if results_drugs.matches:
            knowledge += f"DRUG INFORMATION:\n"
            for i, match in enumerate(results_drugs.matches):
                drug_info = match.metadata
                knowledge += f"--- Drug Match {i+1} (Score: {match.score:.4f}) ---\n"
                knowledge += f"Drug: {drug_info.get('drug', 'N/A')}\n"
                knowledge += f"URL: {drug_info.get('url', 'N/A')}\n"
                knowledge += f"Text: {drug_info.get('text', 'N/A')}\n"
                if 'sections' in drug_info:
                    knowledge += f"Available sections: {', '.join(eval(drug_info['sections']))}\n"
                knowledge += "\n"
        
        return knowledge


    def run_inference(self, input_data, use_openai=False, use_db=False):
        """
        Orchestrates the complete inference process with options for using OpenAI and/or knowledge database.
        
        Args:
            input_data (str): The input query or data to process
            use_openai (bool): Whether to use OpenAI for inference steps
            use_db (bool): Whether to use database for knowledge retrieval
            
        Returns:
            str: The final response after all processing steps
        """
        
        ia_combined_prompt = self.prompt_manager.generate_combined_prompt("inquiry_analysis")
        if use_openai:
            ia_response = self.openai_inference(ia_combined_prompt + input_data)
        else:
            ia_response = self.llama_inference(ia_combined_prompt + input_data, use_soft_prompt=False)

        
        ka_combined_prompt = self.prompt_manager.generate_combined_prompt("knowledge_acquisition")
        
        if use_db:
            
            query = ia_response + " " + ka_combined_prompt + " " + input_data
            dsdg_enriched_input = self.extract_knowledge_db(query)
               
        else:
            
            identified_entities = extract_entities(ia_response)
            ka_response = self.llama_inference(ka_combined_prompt + ia_response, identified_entities)
            dsdg_enriched_input = self.extract_knowledge(ka_response)

        
        eg_combined_prompt = self.prompt_manager.generate_combined_prompt("evidence_generation")
        
        if use_openai:
            
            eg_response = self.openai_inference(eg_combined_prompt+dsdg_enriched_input)
        else:
            eg_response = self.llama_inference(eg_combined_prompt+dsdg_enriched_input, use_soft_prompt=False)

        return eg_response
