import pandas as pd
import re

def preprocess_query(row):
    """
    Preprocess the query to include context and options when available.
    
    Args:
        row: DataFrame row containing the data
        
    Returns:
        str: Processed query with context and options if available
    """
    query = row['question']
    processed_query = query
    
    
    if 'context' in row and pd.notna(row['context']) and row['context']:
        
        if row['context'] not in query:
            processed_query += f"\n\nContext: {row['context']}"
    
    
    option_pattern = r'[A-D]:\s*([^,]+)(?:,|$)'
    if re.search(option_pattern, query):
        
        return processed_query
    
    
    options = {}
    for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
        if opt in row and pd.notna(row[opt]) and row[opt]:
            letter = opt.split('_')[1]
            options[letter] = row[opt]
    
    
    if not options:
        for letter in ['A', 'B', 'C', 'D']:
            if letter in row and pd.notna(row[letter]) and row[letter]:
                options[letter] = row[letter]
    
    
    if not options:
        
        matches = re.findall(r'([A-D])\s*[:)]\s*([^,]+)(?:,|$)', query)
        if matches:
            for letter, option_text in matches:
                options[letter] = option_text.strip()
    
    
    if options:
        processed_query += "\n\nOptions:"
        for letter, option_text in sorted(options.items()):
            processed_query += f"\n{letter}: {option_text}"
    
    return processed_query 