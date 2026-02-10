# Resume Ranker

A resume ranking system that ranks resumes based on job description skills using **Word2Vec** and **NLP**.

## Features

- Extracts skills, experience, email, and phone from PDF resumes
- Uses Word2Vec for semantic similarity matching
- Ranks resumes based on skill match + experience score

## Project Structure

```
resume-ranker/
├── README.md
├── requirements.txt
├── constants.py       # Configuration and paths
├── ranker.py          # Main ranking engine
├── model.py           # Word2Vec model training
├── all_resumes/       # Place PDF resumes here
├── data/              # Corpus, skillsets, stop words
├── model/             # Trained Word2Vec model
└── output/            # Ranked results (CSV/JSON)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Word2Vec Model

```bash
python model.py
```

### 2. Run the Ranker

```bash
python ranker.py
```

## How It Works

1. **PDF Parsing**: Extracts text from resumes using `pdfminer3`
2. **Feature Extraction**: Identifies skills, experience, contact info using NLP
3. **Similarity Scoring**: Computes Word2Vec similarity between job skills and resume skills
4. **Ranking**: `Final Score = Experience × 0.4 + Skill Similarity × 0.6`

## Authors
- Barkha Bharti
- Ashwani Patel  
- Amit Jaiswar
