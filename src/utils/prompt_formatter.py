"""
Utility module for formatting prompts based on question type.
This module provides functions to format prompts for different types of questions:
- Multiple choice questions (A, B, C, D options)
- Binary questions (yes/no answers)
- Text questions (free text responses)
- Double binary questions (two yes/no answers)
"""

def detect_question_type(input_data):
    """
    Detects the type of question based on its content.
    
    Args:
        input_data (str): The input query or data to process
        
    Returns:
        str: The detected question type ('mc', 'binary', 'text', or 'double_binary')
    """
    
    if "\nOptions:" in input_data or "\noptions:" in input_data:
        return "mc"
        
    
    mc_indicators = ["A:", "B:", "C:", "D:", "A.", "B.", "C.", "D.", "A)", "B)", "C)", "D)"]
    if any(indicator in input_data for indicator in mc_indicators):
        return "mc"
        
    
    if "adverse drug reaction" in input_data.lower():
        return "text"
        
    
    if "drug effects" in input_data.lower() or "two questions" in input_data.lower():
        return "double_binary"
        
    
    return "binary"



def format_mc_prompt_cot(prompt):
    """
    Format prompt for multiple-choice questions with Chain of Thought reasoning.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The formatted prompt with instructions for MC response format with analysis
    """
    format_instruction = "\n\nAnalyze the question first, then provide the final answer, which should be a single letter in the alphabet representing the best option among the multiple choices provided in the question. Follow this format:\nAnalysis: (your analysis)\nFinal Answer: (single letter A, B, C, or D)"
    return prompt + format_instruction

def format_binary_prompt_cot(prompt):
    """
    Format prompt for binary yes/no questions with Chain of Thought reasoning.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The formatted prompt with instructions for binary response format with analysis
    """
    format_instruction = "\n\nAnalyze the question first, Provide the final answer, which should be a yes or no. Never say you dont know, always provide a yes or no answer regardless. If you dont have enough information, just guess. You must Follow this format:\nAnalysis: (your analysis)\nFinal Answer: (yes or no)"
    return prompt + format_instruction

def format_text_prompt_cot(prompt):
    """
    Format prompt for free text response questions with Chain of Thought reasoning.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The formatted prompt with instructions for text response format with analysis
    """
    format_instruction = "\n\nAnalyze the question first, then provide the final answer, which should be a single term consisting of english words. Follow this format:\nAnalysis: (your analysis)\nFinal Answer: (concise phrase describing the answer)"
    return prompt + format_instruction

def format_double_binary_prompt_cot(prompt):
    """
    Format prompt for double binary questions with Chain of Thought reasoning.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The formatted prompt with instructions for double binary response format with analysis
    """
    format_instruction = "\n\nYour task is to answer two questions. Analyze the questions first, then provide the final answer, which should consist of two terms, each being either \"yes\" or \"no\" corresponding to the questions. Never say you dont know, always provide a yes or no answer regardless. If you dont have enough information, just guess. Follow this format:\n\nAnalysis: (your analysis)\nFinal Answer: (yes or no), (yes or no)"
    return prompt + format_instruction



def format_mc_prompt_no_cot(prompt):
    """
    Format prompt for multiple-choice questions without Chain of Thought reasoning.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The formatted prompt with instructions for MC response format without analysis
    """
    format_instruction = "\n\nProvide the final answer, which should be a single letter in the alphabet representing the best option among the multiple choices provided in the question. Follow this format:\nFinal Answer: (single letter A, B, C, or D)"
    return prompt + format_instruction

def format_binary_prompt_no_cot(prompt):
    """
    Format prompt for binary yes/no questions without Chain of Thought reasoning.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The formatted prompt with instructions for binary response format without analysis
    """
    format_instruction = "\n\nProvide the final answer, which should be a yes or no. Never say you dont know, always provide a yes or no answer regardless. If you dont have enough information, just guess. Follow this format:\nFinal Answer: (yes or no)"
    return prompt + format_instruction

def format_text_prompt_no_cot(prompt):
    """
    Format prompt for free text response questions without Chain of Thought reasoning.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The formatted prompt with instructions for text response format without analysis
    """
    format_instruction = "\n\nProvide the final answer, which should be a concise phrase describing the answer. Follow this format:\nFinal Answer: (your concise answer)"
    return prompt + format_instruction

def format_double_binary_prompt_no_cot(prompt):
    """
    Format prompt for double binary questions without Chain of Thought reasoning.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The formatted prompt with instructions for double binary response format without analysis
    """
    format_instruction = "\n\nYour task is to answer two questions. Provide the final answer, which should consist of two terms, each being either \"yes\" or \"no\" corresponding to the questions. Follow this format:\nFinal Answer: (yes or no), (yes or no)"
    return prompt + format_instruction



def format_mc_prompt(prompt):
    """Legacy function for backward compatibility. Uses the CoT version."""
    return format_mc_prompt_cot(prompt)

def format_binary_prompt(prompt):
    """Legacy function for backward compatibility. Uses the CoT version."""
    return format_binary_prompt_cot(prompt)

def format_text_prompt(prompt):
    """Legacy function for backward compatibility. Uses the CoT version."""
    return format_text_prompt_cot(prompt)

def format_double_binary_prompt(prompt):
    """Legacy function for backward compatibility. Uses the CoT version."""
    return format_double_binary_prompt_cot(prompt)



def format_prompt(prompt, input_data, use_cot=True):
    """
    Formats a prompt based on the detected question type and whether to use Chain of Thought.
    
    Args:
        prompt (str): The original prompt
        input_data (str): The input query used to detect question type
        use_cot (bool): Whether to use Chain of Thought formatting
        
    Returns:
        str: The formatted prompt with appropriate response format instructions
    """
    question_type = detect_question_type(input_data)
    
    if question_type == "mc":
        return format_mc_prompt_cot(prompt) if use_cot else format_mc_prompt_no_cot(prompt)
    elif question_type == "binary":
        return format_binary_prompt_cot(prompt) if use_cot else format_binary_prompt_no_cot(prompt)
    elif question_type == "double_binary":
        return format_double_binary_prompt_cot(prompt) if use_cot else format_double_binary_prompt_no_cot(prompt)
    else:  
        return format_text_prompt_cot(prompt) if use_cot else format_text_prompt_no_cot(prompt)


FORMATTERS = {
    'mc': {
        'cot': format_mc_prompt_cot,
        'no_cot': format_mc_prompt_no_cot
    },
    'binary': {
        'cot': format_binary_prompt_cot,
        'no_cot': format_binary_prompt_no_cot
    },
    'text': {
        'cot': format_text_prompt_cot,
        'no_cot': format_text_prompt_no_cot
    },
    'double_binary': {
        'cot': format_double_binary_prompt_cot,
        'no_cot': format_double_binary_prompt_no_cot
    }
} 