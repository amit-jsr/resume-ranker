"""
----------------------------------------------------------------------------------------------------------------------
List of Constants used in ranker.py and model.py
----------------------------------------------------------------------------------------------------------------------
Created on: 22 June 2020
---------------------------------------------------------------------------------------------------------------------
"""
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Data paths
CORPUS_PATH = os.path.join(BASE_PATH, 'data', 'corpus.txt')
STOP_WORD_PATH = os.path.join(BASE_PATH, 'data', 'stop_words.txt')
SKILL_SET_PATH = os.path.join(BASE_PATH, 'data', 'skillsets.csv')
MUST_SKILLS_PATH = os.path.join(BASE_PATH, 'data', 'must_skills.csv')
USER_DATA_PATH = os.path.join(BASE_PATH, 'data', 'user_data.csv')

# Model paths
MODEL_PATH = os.path.join(BASE_PATH, 'model', 'word2vec.model')

# Output paths
OUTPUT_CSV_PATH = os.path.join(BASE_PATH, 'output', 'top_resume.csv')
OUTPUT_JSON_PATH = os.path.join(BASE_PATH, 'output', 'top_resume.json')
SKILL_SCORE_CSV_PATH = os.path.join(BASE_PATH, 'output', 'skill_score.csv')
SKILL_SCORE_JSON_PATH = os.path.join(BASE_PATH, 'output', 'skill_score.json')

# Resume directory
RESUME_DIR_PATH = os.path.join(BASE_PATH, 'all_resumes')

# Regex patterns
EMAIL_REGEX = r"([^@|\s]+@[^@]+\.[^@|\s]+)"
MOBILE_REGEX = (
    r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?'
    r'(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|'
    r'([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?'
    r'([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?'
    r'([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'
)
