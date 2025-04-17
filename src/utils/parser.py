def binary_parser(output):
    sections = output.split('\n')
    analysis = ""
    final_answer = ""
    for section in sections:
        section_lower = section.lower()
        if section_lower.startswith("analysis: "):
            analysis = section_lower.replace("analysis: ", "")
        elif section_lower.startswith("final answer: "):
            final_answer = section_lower.replace("final answer: ", "").strip()
    
    if 'yes' in final_answer:
        final_answer = 'yes'
    elif 'no' in final_answer:
        final_answer = 'no'
    else:
        final_answer = ""
    return analysis, final_answer


def mc_parser(output):
    sections = output.split('\n')
    analysis = ""
    final_answer = ""
    for section in sections:
        section_lower = section.lower()
        if section_lower.startswith("analysis: "):
            analysis = section_lower.replace("analysis: ", "")
        elif section_lower.startswith("final answer: "):
            final_answer = section_lower.replace("final answer: ", "").strip()
    
    final_answer = final_answer[0] if final_answer and final_answer[0].isalpha() else ""
    return analysis, final_answer


def text_parser(output):
    sections = output.split('\n')
    analysis = ""
    final_answer = ""
    for section in sections:
        section_lower = section.lower()
        if section_lower.startswith("analysis: "):
            analysis = section_lower.replace("analysis: ", "")
        elif section_lower.startswith("final answer: "):
            final_answer = section_lower.replace("final answer: ", "").strip()
    return analysis, final_answer


def double_binary_parser(output):
    """
    Parser for double binary responses in the form "yes, no" or similar.
    
    Args:
        output (str): The model output to parse
        
    Returns:
        tuple: (analysis, final_answer) where final_answer is in the form "yes, no"
    """
    sections = output.split('\n')
    analysis = ""
    final_answer = ""
    
    for section in sections:
        section_lower = section.lower()
        if section_lower.startswith("analysis: "):
            analysis = section_lower.replace("analysis: ", "")
        elif section_lower.startswith("final answer: "):
            final_answer = section_lower.replace("final answer: ", "").strip()
    
    
    answer_parts = final_answer.split(',')
    if len(answer_parts) == 2:
        first_answer = 'yes' if 'yes' in answer_parts[0].lower() else 'no'
        second_answer = 'yes' if 'yes' in answer_parts[1].lower() else 'no'
        final_answer = f"{first_answer}, {second_answer}"
    else:
        
        final_answer = "no, no"
    
    return analysis, final_answer
