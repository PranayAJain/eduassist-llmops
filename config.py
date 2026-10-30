import os
from openai import OpenAI

BASE_URL   = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL      = "gemini-3.5-flash-lite"
API_STYLE  = "Gemini API v1beta via OpenAI-compatible endpoint"
client     = OpenAI(base_url=BASE_URL, api_key=os.environ["GEMINI_API_KEY"])

PRICE_DATE         = "2026-10-30"
CURRENCY           = "USD"
INPUT_PRICE_PER_M  = 0.30
OUTPUT_PRICE_PER_M = 2.50
MONTHLY_REQUESTS   = 10_000

QUESTIONS = {
    "Q1": "What is overfitting in machine learning and how can it be prevented?",
    "Q2": "Explain the difference between supervised and unsupervised learning.",
    "Q3": "What is the time complexity of binary search and why?",
    "Q4": "What is the role of a primary key in a relational database?",
    "Q5": "Explain the concept of gradient descent in simple terms.",
}
