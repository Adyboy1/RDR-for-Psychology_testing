# llm_api.py
import google.generativeai as genai
import os
import json
import logging
from typing import Optional, List

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
MODEL_NAME = 'gemini-2.5-flash'
# Prompt Files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to PSYCHOLOGY_RDR, then into data/prompts
PROMPTS_DIR = os.path.join(CURRENT_DIR, '..','prompts')

# Define full paths for prompt files
CHECK_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'prompt_condition.txt')
DIFF_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'prompt_differentiate.txt')
SUMMARY_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'prompt_summary.txt')
MERGE_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'prompt_merge.txt')

# --- Setup ---
try:
    api_key = os.environ["API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    raise EnvironmentError(f"Setup failed: {e}")

# --- Load Prompts ---
def load_prompt(filename):
    try:
        with open(filename, "r") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        return ""

CHECK_TEMPLATE = load_prompt(CHECK_PROMPT_FILE)
DIFF_TEMPLATE = load_prompt(DIFF_PROMPT_FILE)
SUMMARY_TEMPLATE = load_prompt(SUMMARY_PROMPT_FILE)
MERGE_TEMPLATE = load_prompt(MERGE_PROMPT_FILE)

# --- Core Functions ---

def llm_generate_summary(transcript_text: str) -> str:
    """
    Reads raw docx transcript and generates a Clinical Prototype Summary.
    """
    try:
        prompt = SUMMARY_TEMPLATE.format(transcript_content=transcript_text)
        #need to finetune these values
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 1.0, "top_p": 1.0} 
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Summary generation failed: {e}")
        return ""

def llm_check_condition(summary_text: str, condition_string: str) -> bool:
    """
    Checks if a SUMMARY satisfies a condition.
    """
    prompt = CHECK_TEMPLATE.format(
        summary_content=summary_text, 
        condition_string=condition_string
    )
    try: #temperature and top p being modified
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 1.0, "top_p": 1.0} 
        )
        logging.info(f"LLM Check: '{condition_string}' -> {response.text.strip().upper()}")
        return 'TRUE' in response.text.strip().upper()
    except Exception:
        return False

def llm_merge_summaries(old_summary: str, new_summary: str) -> str:
    """
    Merges a new patient profile into an existing consolidated group profile.
    """
    prompt = MERGE_TEMPLATE.format(
        old_summary=old_summary,
        new_summary=new_summary
    )
    try: # need to finetune these values
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 1.0, "top_p": 1.0} 
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Merge failed: {e}")
        return old_summary # Fallback: keep old summary

def llm_get_differentiating_conditions(new_summary: str, ref_summary: str) -> List[str]:
    """
    Compares NEW summary vs REFERENCE summary to find differences.
    """
    prompt = DIFF_TEMPLATE.format(
        summary_new=new_summary,
        summary_ref=ref_summary if ref_summary else "None"
    )
    
    try: #top p temperature added
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 1.0, 
                "top_p": 1.0,
            }
        )
        text = response.text.strip()
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
        
        return json.loads(text)
    except Exception as e:
        logging.error(f"Diff extraction failed: {e}")
        return []
    
def llm_verify_manual_condition(summary_text: str, condition_string: str) -> bool:
    """
    Verifies if a manually entered condition is TRUE for a given summary.
    Used to ensure the Reference Summary does NOT meet the condition.
    """
    # Uses the same logic as check_condition but exposed for specific manual verification tasks
    prompt = CHECK_TEMPLATE.format(
        summary_content=summary_text, 
        condition_string=condition_string
    )
    try: 
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 1.0, "top_p": 1.0} 
        )
        result = response.text.strip().upper()
        logging.info(f"Manual Verification: '{condition_string}' -> {result}")
        return 'TRUE' 
    except Exception as e:
        logging.error(f"Manual verification failed: {e}")
        # Default to False on error to be safe
        return False