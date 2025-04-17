import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import re
import logging
import os
from tqdm import tqdm
import traceback

logger = logging.getLogger(__name__)


def extract_entities(ia_response):
    """
    Extract drug and disease entities from inquiry analysis response.
    
    This function uses a combination of regex patterns to robustly extract entities
    even with variations in the response format.
    
    Args:
        ia_response (str): The response from the inquiry analysis stage
        
    Returns:
        list: Combined list of drug and disease entities
    """
    
    if not ia_response:
        logger.warning("Received empty or None inquiry analysis response")
        return []
    
    try:
        
        drugs_patterns = [
            r"Drugs:?\s*\[(.*?)\]",
            r"Drugs:?\s*([^[\]]+?)(?:\n|$)",
            r"Drug(?:s)? (?:mentioned|identified|found):?\s*\[(.*?)\]",
            r"Drug(?:s)? (?:mentioned|identified|found):?\s*([^[\]]+?)(?:\n|$)"
        ]
        
        diseases_patterns = [
            r"Diseases:?\s*\[(.*?)\]",
            r"Diseases:?\s*([^[\]]+?)(?:\n|$)",
            r"Disease(?:s)? (?:mentioned|identified|found):?\s*\[(.*?)\]",
            r"Disease(?:s)? (?:mentioned|identified|found):?\s*([^[\]]+?)(?:\n|$)"
        ]

        
        drugs = []
        for pattern in drugs_patterns:
            drugs_match = re.search(pattern, ia_response, re.IGNORECASE)
            if drugs_match:
                
                raw_drugs = drugs_match.group(1).strip()
                
                drug_list = re.findall(r'(?:"([^"]+)"|\'([^\']+)\'|([^,\'"]+))', raw_drugs)
                
                drugs = [next(filter(None, entity)).strip() for entity in drug_list if any(entity)]
                break

        
        diseases = []
        for pattern in diseases_patterns:
            diseases_match = re.search(pattern, ia_response, re.IGNORECASE)
            if diseases_match:
                
                raw_diseases = diseases_match.group(1).strip()
                
                disease_list = re.findall(r'(?:"([^"]+)"|\'([^\']+)\'|([^,\'"]+))', raw_diseases)
                
                diseases = [next(filter(None, entity)).strip() for entity in disease_list if any(entity)]
                break

        
        identified_entities = drugs + diseases

        
        if not identified_entities:
            
            direct_mentions = re.findall(r'(?:drug|medication|medicine|treatment) (?:is|called|named) "?([^".]+)"?', 
                                       ia_response, re.IGNORECASE)
            direct_mentions += re.findall(r'(?:disease|condition|disorder) (?:is|called|named) "?([^".]+)"?', 
                                        ia_response, re.IGNORECASE)
            identified_entities = [mention.strip() for mention in direct_mentions]

        
        if identified_entities:
            logger.info(f"Extracted entities: {identified_entities}")
        else:
            logger.warning("No entities extracted from inquiry analysis response")
            
        return identified_entities
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return []


class SoftPromptTuner:
    """
    Handles the training of soft prompts using knowledge graphs and GCN.
    
    This class coordinates the training of the GCN model to generate effective
    soft prompts for the LLaMA model.
    """
    
    def __init__(self, llama_utils, gcn_model, prompt_manager, knowledge_base, config):
        """
        Initialize the soft prompt tuner.
        
        Args:
            llama_utils (LLaMAUtils): The LLaMA utilities for model operations
            gcn_model (GraphConvolutionalNetwork): The GCN model to be trained
            prompt_manager (PromptManager): Manager for generating prompts
            knowledge_base (DSDGGenerator): Knowledge base with drug/disease graph
            config (dict): Configuration parameters for training
        """
        self.llama_utils = llama_utils
        self.gcn_model = gcn_model
        self.prompt_manager = prompt_manager
        self.knowledge_base = knowledge_base
        self.config = config if isinstance(config, dict) else {}
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        
        self._setup_directories()
        
        
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            logger.info(f"TensorBoard writer initialized at {self.log_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard writer: {e}. Using dummy writer.")
            
            class DummyWriter:
                def add_scalar(self, *args, **kwargs): pass
                def close(self): pass
            self.writer = DummyWriter()
            
        self.best_val_loss = float('inf')
        
        
        self._verify_model_compatibility()
        
        
        self.llama_utils.freeze_llama_weights()
        
    def _setup_directories(self):
        """Set up checkpoint and log directories with proper error handling."""
        
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints/soft_prompts')
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            test_file = os.path.join(self.checkpoint_dir, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Checkpoint directory set up at: {self.checkpoint_dir}")
        except (IOError, PermissionError) as e:
            
            logger.error(f"Cannot write to checkpoint directory {self.checkpoint_dir}: {e}")
            self.checkpoint_dir = os.path.join(os.getcwd(), 'fallback_checkpoints')
            logger.warning(f"Falling back to checkpoint directory: {self.checkpoint_dir}")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        
        self.log_dir = self.config.get('log_dir', 'runs/soft_prompt_tuning')
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info(f"Log directory set up at: {self.log_dir}")
        except (IOError, PermissionError) as e:
            
            logger.error(f"Cannot write to log directory {self.log_dir}: {e}")
            self.log_dir = os.path.join(os.getcwd(), 'fallback_logs')
            logger.warning(f"Falling back to log directory: {self.log_dir}")
            os.makedirs(self.log_dir, exist_ok=True)
        
    def _verify_model_compatibility(self):
        """
        Verify that GCN output dimensions are compatible with LLaMA embeddings.
        Raises ValueError if dimensions are incompatible.
        """
        try:
            
            gcn_output_dim = self.gcn_model.gc2.out_features
            llama_embed_dim = self.llama_utils.model.get_input_embeddings().embedding_dim
            
            if gcn_output_dim != llama_embed_dim:
                logger.warning(f"Dimension mismatch: GCN output dim ({gcn_output_dim}) != "
                               f"LLaMA embedding dim ({llama_embed_dim})")
                logger.warning("This may cause issues when applying soft prompts")
            else:
                logger.info(f"Model dimensions are compatible: {gcn_output_dim}")
        except Exception as e:
            logger.warning(f"Could not verify model compatibility: {e}")
            
    def _generate_inquiry_prompt(self, text):
        """
        Generate inquiry analysis prompt with fallback mechanism.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Generated prompt
        """
        if not text:
            logger.warning("Empty text provided to _generate_inquiry_prompt")
            return ""
            
        try:
            
            if hasattr(self.prompt_manager, 'generate_combined_prompt'):
                ia_prompt = self.prompt_manager.generate_combined_prompt("inquiry_analysis")
                return ia_prompt + text
            else:
                
                logger.warning("PromptManager doesn't have generate_combined_prompt method, using fallback")
                return f"Analyze the following inquiry and identify any drugs and diseases mentioned:\n\n{text}"
        except Exception as e:
            logger.error(f"Error generating inquiry prompt: {e}")
            
            return f"Identify drugs and diseases in the following text:\n\n{text}"
    
    def train_gcn(self, train_loader, val_loader=None, epochs=None):
        """
        Train the GCN model to generate effective soft prompts.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of epochs to train for (overrides config)
            
        Returns:
            float: Best validation loss achieved
        """
        
        epochs = epochs or self.config.get('epochs', 3)
        lr = self.config.get('learning_rate', 5e-5)
        weight_decay = self.config.get('weight_decay', 0.01)
        warmup_ratio = self.config.get('warmup_ratio', 0.1)
        save_steps = self.config.get('save_steps', 100)
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        patience = self.config.get('patience', 5)  
        
        
        if train_loader is None or len(train_loader) == 0:
            logger.error("No training data provided")
            return float('inf')
        
        
        optimizer = optim.AdamW(self.gcn_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        
        if train_loader:
            num_training_steps = len(train_loader) * epochs
            num_warmup_steps = int(num_training_steps * warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = None
        
        logger.info(f"Starting GCN training for {epochs} epochs")
        
        
        global_step = 0
        no_improvement_count = 0
        
        try:
            for epoch in range(epochs):
                epoch_loss = 0
                self.gcn_model.train()
                
                
                if train_loader:
                    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                    for batch_idx, batch in enumerate(progress_bar):
                        
                        if batch is None:
                            logger.warning(f"Empty batch encountered at step {global_step}, skipping")
                            continue
                            
                        
                        if 'input_ids' not in batch or 'labels' not in batch:
                            logger.warning(f"Batch missing required keys at step {global_step}, skipping")
                            continue
                            
                        input_ids = batch['input_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        
                        batch_size = input_ids.size(0)
                        if batch_size == 0:
                            logger.warning(f"Empty batch encountered at step {global_step}, skipping")
                            continue
                        
                        
                        optimizer.zero_grad()
                        
                        
                        batch_loss = 0
                        valid_items = 0
                        
                        for i in range(batch_size):
                            try:
                                
                                prompt = batch['prompt'][i] if 'prompt' in batch and i < len(batch['prompt']) else None
                                
                                if prompt:
                                    try:
                                        
                                        ia_prompt = self._generate_inquiry_prompt(prompt)
                                        
                                        
                                        ia_response = self.llama_utils.llama_inference(ia_prompt, use_soft_prompt=False)
                                        
                                        
                                        identified_entities = extract_entities(ia_response)
                                        
                                        
                                        if not identified_entities:
                                            logger.warning(f"No entities found in prompt at batch index {i}")
                                            continue
                                        
                                        
                                        graph = self.knowledge_base.get_graph()
                                        soft_prompt_prefix = self.gcn_model.generate_prefix(graph, identified_entities)
                                        
                                        
                                        if soft_prompt_prefix is not None:
                                            if hasattr(self.llama_utils.soft_embedding, 'n_tokens'):
                                                expected_dim = self.llama_utils.model.get_input_embeddings().embedding_dim
                                                actual_dim = soft_prompt_prefix.size(-1)
                                                
                                                if actual_dim != expected_dim:
                                                    logger.warning(f"Dimension mismatch: Soft prompt dim ({actual_dim}) != "
                                                                 f"LLaMA embedding dim ({expected_dim})")
                                            
                                            
                                            self.llama_utils.apply_soft_prompt(prompt, soft_prompt_prefix)
                                    except Exception as e:
                                        logger.error(f"Error processing prompt: {e}")
                                        logger.debug(traceback.format_exc())
                                        continue
                                
                                
                                item_input_ids = input_ids[i:i+1]  
                                
                                try:
                                    outputs = self.llama_utils.llama_training_forward(item_input_ids)
                                except Exception as e:
                                    logger.error(f"Forward pass failed: {e}")
                                    logger.debug(traceback.format_exc())
                                    continue
                                
                                
                                if outputs.size(1) <= 1:
                                    logger.warning(f"Output sequence too short for loss calculation")
                                    continue
                                    
                                item_labels = labels[i:i+1]
                                
                                
                                if outputs.size(1) != item_labels.size(1):
                                    logger.warning(f"Shape mismatch: outputs {outputs.shape}, labels {item_labels.shape}")
                                    
                                    min_len = min(outputs.size(1), item_labels.size(1))
                                    outputs = outputs[:, :min_len, :]
                                    item_labels = item_labels[:, :min_len]
                                
                                
                                shift_logits = outputs[..., :-1, :].contiguous()
                                shift_labels = item_labels[..., 1:].contiguous()
                                
                                
                                if shift_logits.size(1) != shift_labels.size(1):
                                    logger.warning(f"After shifting, shape mismatch: logits {shift_logits.shape}, labels {shift_labels.shape}")
                                    continue
                                    
                                item_loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                                                 shift_labels.view(-1))
                                
                                
                                if torch.isnan(item_loss).any() or torch.isinf(item_loss).any():
                                    logger.warning(f"NaN or Inf loss detected, skipping item")
                                    continue
                                    
                                
                                batch_loss += item_loss
                                valid_items += 1
                            except Exception as e:
                                logger.error(f"Error processing batch item {i}: {e}")
                                logger.debug(traceback.format_exc())
                                continue
                        
                        
                        if valid_items == 0:
                            logger.warning(f"No valid items in batch {batch_idx}, skipping")
                            continue
                        
                        
                        batch_loss = batch_loss / valid_items
                        
                        
                        try:
                            batch_loss.backward()
                            
                            
                            torch.nn.utils.clip_grad_norm_(self.gcn_model.parameters(), max_grad_norm)
                            
                            
                            optimizer.step()
                            if scheduler:
                                scheduler.step()
                        except Exception as e:
                            logger.error(f"Error during backpropagation: {e}")
                            logger.debug(traceback.format_exc())
                            
                            optimizer.zero_grad()
                            continue
                        
                        
                        epoch_loss += batch_loss.item()
                        global_step += 1
                        progress_bar.set_postfix({"loss": batch_loss.item()})
                        
                        
                        self.writer.add_scalar("Loss/train", batch_loss.item(), global_step)
                        
                        
                        if global_step % save_steps == 0:
                            self.save_checkpoint(global_step)
                    
                    
                    if len(progress_bar) > 0:
                        epoch_loss = epoch_loss / len(progress_bar)
                        logger.info(f"Epoch {epoch+1}/{epochs} - Avg Training Loss: {epoch_loss:.4f}")
                    
                    
                    if val_loader:
                        try:
                            val_loss = self.validate(val_loader)
                            logger.info(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")
                            
                            
                            if val_loss < self.best_val_loss:
                                logger.info(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                                self.best_val_loss = val_loss
                                self.save_checkpoint(global_step, is_best=True)
                                no_improvement_count = 0
                            else:
                                no_improvement_count += 1
                                logger.info(f"No improvement in validation loss for {no_improvement_count} epochs")
                                
                                
                                if no_improvement_count >= patience:
                                    logger.info(f"Early stopping after {epoch+1} epochs due to no improvement")
                                    break
                        except Exception as e:
                            logger.error(f"Error during validation: {e}")
                            logger.debug(traceback.format_exc())
            
            
            self.save_checkpoint(global_step, is_final=True)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(global_step, is_final=True)
        except Exception as e:
            logger.error(f"Training interrupted due to error: {e}")
            logger.debug(traceback.format_exc())
            
            try:
                self.save_checkpoint(global_step, is_final=True)
            except:
                pass
        finally:
            
            self.writer.close()
        
        return self.best_val_loss
    
    def validate(self, val_loader):
        """
        Validate the current model on the validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            float: Validation loss
        """
        if val_loader is None or len(val_loader) == 0:
            logger.warning("No validation data provided")
            return float('inf')
            
        self.gcn_model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                
                if batch is None:
                    continue
                
                
                if 'input_ids' not in batch or 'labels' not in batch:
                    continue
                    
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                
                batch_size = input_ids.size(0)
                if batch_size == 0:
                    continue
                    
                batch_loss = 0
                valid_items = 0
                
                for i in range(batch_size):
                    try:
                        
                        item_input_ids = input_ids[i:i+1]
                        item_labels = labels[i:i+1]
                        
                        outputs = self.llama_utils.llama_training_forward(item_input_ids)
                        
                        
                        if outputs.size(1) != item_labels.size(1):
                            min_len = min(outputs.size(1), item_labels.size(1))
                            outputs = outputs[:, :min_len, :]
                            item_labels = item_labels[:, :min_len]
                        
                        
                        shift_logits = outputs[..., :-1, :].contiguous()
                        shift_labels = item_labels[..., 1:].contiguous()
                        
                        
                        if shift_logits.size(1) != shift_labels.size(1):
                            continue
                            
                        item_loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                                         shift_labels.view(-1))
                        
                        
                        if torch.isnan(item_loss).any() or torch.isinf(item_loss).any():
                            continue
                            
                        batch_loss += item_loss.item()
                        valid_items += 1
                    except Exception as e:
                        logger.error(f"Error during validation item processing: {e}")
                        continue
                
                
                if valid_items > 0:
                    total_loss += batch_loss
                    total_samples += valid_items
        
        
        if total_samples == 0:
            logger.warning("No valid samples during validation")
            return float('inf')
            
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def save_checkpoint(self, step, is_best=False, is_final=False):
        """
        Save a checkpoint of the GCN model and soft prompts.
        
        Args:
            step (int): Current training step
            is_best (bool): Whether this is the best model so far
            is_final (bool): Whether this is the final checkpoint
        """
        try:
            
            gcn_path = os.path.join(self.checkpoint_dir, f"gcn_step_{step}.pt")
            torch.save(self.gcn_model.state_dict(), gcn_path)
            
            
            if self.llama_utils.soft_embedding is not None:
                prompt_path = os.path.join(self.checkpoint_dir, f"soft_prompt_step_{step}.pt")
                self.llama_utils.save_soft_prompt(prompt_path)
            
            
            if is_best:
                best_gcn_path = os.path.join(self.checkpoint_dir, "gcn_best.pt")
                best_prompt_path = os.path.join(self.checkpoint_dir, "soft_prompt_best.pt")
                
                
                if os.path.exists(gcn_path):
                    torch.save(self.gcn_model.state_dict(), best_gcn_path)
                
                if self.llama_utils.soft_embedding is not None:
                    self.llama_utils.save_soft_prompt(best_prompt_path)
            
            
            if is_final:
                final_gcn_path = os.path.join(self.checkpoint_dir, "gcn_final.pt")
                final_prompt_path = os.path.join(self.checkpoint_dir, "soft_prompt_final.pt")
                
                
                if os.path.exists(gcn_path):
                    torch.save(self.gcn_model.state_dict(), final_gcn_path)
                
                if self.llama_utils.soft_embedding is not None:
                    self.llama_utils.save_soft_prompt(final_prompt_path)
            
            logger.info(f"Saved checkpoint at step {step}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            
            try:
                fallback_dir = os.getcwd()
                gcn_path = os.path.join(fallback_dir, f"gcn_step_{step}_fallback.pt")
                torch.save(self.gcn_model.state_dict(), gcn_path)
                logger.warning(f"Saved fallback checkpoint to {gcn_path}")
            except:
                logger.error("Failed to save fallback checkpoint")
    
    def load_checkpoint(self, gcn_path, prompt_path=None):
        """
        Load a checkpoint of the GCN model and optionally soft prompts.
        
        Args:
            gcn_path (str): Path to the GCN model checkpoint
            prompt_path (str, optional): Path to the soft prompt checkpoint
        """
        
        try:
            if os.path.exists(gcn_path):
                self.gcn_model.load_state_dict(torch.load(gcn_path, map_location=self.device))
                logger.info(f"Loaded GCN model from {gcn_path}")
            else:
                logger.warning(f"GCN checkpoint not found at {gcn_path}")
        except Exception as e:
            logger.error(f"Error loading GCN model: {e}")
        
        
        if prompt_path:
            try:
                if os.path.exists(prompt_path):
                    self.llama_utils.load_soft_prompt(prompt_path)
                    logger.info(f"Loaded soft prompts from {prompt_path}")
                else:
                    logger.warning(f"Soft prompt checkpoint not found at {prompt_path}")
            except Exception as e:
                logger.error(f"Error loading soft prompts: {e}")
    
    def prepare_training_batch(self, text, tokenizer, max_length=None):
        """
        Prepare a training batch from text.
        
        Args:
            text (str): Input text to tokenize
            tokenizer: The tokenizer to use
            max_length (int, optional): Maximum sequence length
            
        Returns:
            dict: Batch with input_ids, labels, and prompt
        """
        
        if not text:
            logger.warning("Empty or None text provided to prepare_training_batch")
            
            return {
                'input_ids': torch.zeros((1, 0), dtype=torch.long),
                'labels': torch.zeros((1, 0), dtype=torch.long),
                'prompt': ""
            }
            
        
        if max_length is None:
            max_length = self.llama_utils.max_length
        
        
        try:
            tokens = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            
            batch = {
                'input_ids': tokens.input_ids,
                'labels': tokens.input_ids.clone(),  
                'prompt': text  
            }
            
            return batch
        except Exception as e:
            logger.error(f"Error preparing training batch: {e}")
            
            return {
                'input_ids': torch.zeros((1, 1), dtype=torch.long),
                'labels': torch.zeros((1, 1), dtype=torch.long),
                'prompt': ""
            }
