import os
import pickle

from dotenv import load_dotenv

load_dotenv()

# pkl file load
with open('./resource/final_result.pkl', 'rb') as file:
    faq_data = pickle.load(file)