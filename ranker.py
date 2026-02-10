# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------------------------------------------------
Resume Ranker
----------------------------------------------------------------------------------------------------------------------
Created on: 22 June 2020
---------------------------------------------------------------------------------------------------------------------
"""

import os
import io
import regex as re
import pandas as pd
import nltk
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from gensim.models.word2vec import Word2Vec
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer

import constants

# Download NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class Preprocessor:
    """Pre-processing the document text."""
    
    def __init__(self):
        """Load the set of stop words from file."""
        with open(constants.STOP_WORD_PATH, 'rb') as file:
            stop = str(file.read())
        self.stop_words = stop.split()

    def tokenize(self, data):
        """Tokenization of text.
        
        Args:
            data: List of token objects with .text attribute
            
        Returns:
            token_list: list of tokens
        """
        token_list = [token.text for token in data]
        return token_list

    def refine_doc(self, data):
        """ Refinement of document text
        Parameters-
            data: document text(string)
        Return-
            filtered_words: list of tokens after text pre-processing and removing unwanted words
        """
        text = data.lower()
        text = text.replace('{html}', "")
        rem_url = re.sub(r'http\S+', '', text)
        encoded_string = rem_url.encode("ascii", "ignore")
        rem_url = encoded_string.decode()
        tokenizer = RegexpTokenizer("[a-zA-Z0-9_#+.]+")
        tokens = tokenizer.tokenize(rem_url)
        filtered_words = [w for w in tokens if w not in self.stop_words]
        filtered_words = list(set(filtered_words))
        return filtered_words


class Ranker:
    """Resume Ranker using content based filtering.
    
    Features:
        1. Skills
        2. Experience
    """
    
    def __init__(self, skill_set_path, directory, model):
        """Initialize the recommender.
        
        Args:
            skill_set_path: path of the universal skillset
            directory: path of the directory containing all the resumes
            model: Word2vec model object
        """
        self.skill_path = skill_set_path
        self.directory = directory
        self.model = model
        self.preprocessor = Preprocessor()
        self.df_user = pd.DataFrame(columns=['file_name', 'mobile', 'email', 'experience', 'skills'])
        self.skills = []
        self.vocab = list(self.model.wv.key_to_index.keys())
        self.vocab.sort()
        self._load_skillset()
        self._parse_all_pdfs()
        self.df_user.to_csv(constants.USER_DATA_PATH)

    def _load_skillset(self):
        """Populate the skills list with skills from the dataset."""
        data = pd.read_csv(self.skill_path, usecols=['Example'])
        data.drop_duplicates(subset='Example', inplace=True)
        data.reset_index(drop=True, inplace=True)
        self.skills = list(data['Example'])

    def convert_pdf_to_txt(self, path):
        """ convert the pdf data to text
        Parameter:
            path: path of the pdf file
        Return:
            text in string format
        """
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        with open(path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()

        # close open handles
        converter.close()
        fake_file_handle.close()
        return text

    def extract_experience(self, data):
        """ Extracted the experience if the word 'experience' present in the sentence
        Parameter:
            data: document text in string format
        Return:
            exp: experience
        """
        lines = [wholeline.strip() for wholeline in data.split("\n") if len(wholeline) > 0]
        lines = [nltk.word_tokenize(wholeline) for wholeline in lines]
        lines = [nltk.pos_tag(wholeline) for wholeline in lines]

        for sentence in lines:
            sen = " ".join([words[0].lower() for words in sentence])
            if re.search('experience', sen):
                sen_tokenized = nltk.word_tokenize(sen)
                tagged = nltk.pos_tag(sen_tokenized)
                entities = nltk.chunk.ne_chunk(tagged)
                for subtree in entities.subtrees():
                    for leaf in subtree.leaves():
                        if leaf[1] == 'CD':
                            exp = leaf[0]
                            exp = re.sub('[^0-9.]+', '', exp)
                            return exp

        return 0

    def extract_mobile(self, data):
        """Extract phone number using regular expression.
        
        Args:
            data: document text in string format
            
        Returns:
            Phone number string or None
        """
        phone = re.findall(re.compile(constants.MOBILE_REGEX), data)
        if phone:
            number = ''.join(phone[0])
            if len(number) > 10:
                return '+' + number
            else:
                return number
        return None

    def extract_email(self, text):
        """Extract email-id using regular expression.
        
        Args:
            text: document text in string format
            
        Returns:
            Email string or None
        """
        email = re.findall(constants.EMAIL_REGEX, text)
        if email:
            try:
                return email[0].split()[0].strip(';')
            except IndexError:
                return None
        return None

    def extract_skills(self, data):
        """Extract skills having semantic similarity with skills list.
        
        Threshold for similarity: 0.75 or more
        Ngrams: up to 4 grams
        
        Args:
            data: document text in string format
            
        Returns:
            skill_set: comma-separated string of extracted skills
        """
        tokens = self.preprocessor.refine_doc(data)
        skill_set = []
        for n in range(1, 5):
            n_grams = ngrams(tokens, n)
            lists = [' '.join(grams) for grams in n_grams]
            for word in lists:
                if word in self.vocab:
                    for s in self.skills:
                        try:
                            if word == s:
                                skill_set.append(word)
                                break
                            elif self.model.wv.similarity(word, s) > 0.75:
                                skill_set.append(word)
                                break
                        except KeyError:
                            continue
        return ','.join(list(set(skill_set)))

    def _parse_all_pdfs(self):
        """Parse all resumes in PDF format in the resume directory.
        
        - Convert PDF to text format
        - Extract phone number, email-id, experience and skills set
        - Append extracted results into dataframe
        """
        results = []
        for i, filename in enumerate(os.listdir(self.directory), 1):
            if filename.endswith(".pdf"):
                try:
                    data = self.convert_pdf_to_txt(os.path.join(self.directory, filename))
                    mobile = self.extract_mobile(data)
                    email = self.extract_email(data)
                    exp = self.extract_experience(data)
                    skills = self.extract_skills(data)
                    print(f'Parsed: {i}')
                    results.append({
                        'file_name': filename,
                        'mobile': mobile,
                        'email': email,
                        'skills': skills,
                        'experience': float(exp) if exp else 0.0
                    })
                except Exception as e:
                    print(f'Error parsing {filename}: {e}')
                    continue
        
        if results:
            self.df_user = pd.concat([self.df_user, pd.DataFrame(results)], ignore_index=True)

    def rank(self, query, top):
        """Rank resumes based on query (desired skills) and output top resumes.
        
        Description:
            - Find similarity score of resume skill set with each desired skill
            - Final Score = Experience * 0.4 + Similarity Score * 0.6
            - Rank resumes based on Final score and store in CSV and JSON format
            
        Args:
            query: desired skills (comma-separated string)
            top: Number of resumes to return in final output
        """
        desired_skills = [str(item).strip() for item in query.split(',')]
        score_dict = {}
        df_result = self.df_user.copy()
        
        for idx in df_result.index:
            score = 0
            temp_dict = {}
            for desired_skill in desired_skills:
                if desired_skill in self.vocab:
                    resume_skills = df_result.iloc[idx]['skills']
                    if resume_skills:
                        for resume_skill in resume_skills.split(','):
                            try:
                                similarity = self.model.wv.similarity(desired_skill, resume_skill)
                                temp_dict[(desired_skill, resume_skill)] = similarity
                            except KeyError:
                                continue
            
            for _ in range(len(desired_skills)):
                try:
                    key_max = max(temp_dict, key=temp_dict.get)
                except ValueError:
                    continue
                score += temp_dict[key_max]
                for temp in temp_dict:
                    if temp[0] == key_max[0] or temp[1] == key_max[1]:
                        temp_dict[temp] = 0
            score_dict[idx] = score
    
        df_result['similarity_score'] = list(score_dict.values())
        df_result['final_score'] = df_result['experience'] * 0.4 + df_result['similarity_score'] * 0.6
        df_result.sort_values(by='final_score', ascending=False, inplace=True, ignore_index=True)
        df_result.iloc[:top].to_csv(constants.OUTPUT_CSV_PATH, index=False)
        df_result.iloc[:top].to_json(constants.OUTPUT_JSON_PATH)


def main():
    """Main entry point for the ranker."""
    model = Word2Vec.load(constants.MODEL_PATH)
    print('Successfully loaded the model!')
    ranker = Ranker(constants.MUST_SKILLS_PATH, constants.RESUME_DIR_PATH, model)
    
    # Uncomment below for interactive mode
    # top = int(input('\nEnter the value of top to find the top resumes: '))
    # while True:
    #     query = input("\nEnter desired skills (comma separated): ")
    #     ranker.rank(query, top)
    #     if input('\nWant to check for more query (Y/N): ').upper() == 'N':
    #         break


if __name__ == "__main__":
    main()

