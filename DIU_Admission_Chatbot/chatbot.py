import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import faqs
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)
questions = []
for faq in faqs:
    questions.append(preprocess(faq["question"]))
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

print("=" * 60)
print("       DIU ADMISSION INFORMATION CHATBOT")
print("=" * 60)
print("Ask your admission-related question.")
print("Type 'exit' to close the chatbot.")
while True:
    user_question = input("\nYou: ")
    if user_question.lower() == "exit":
        print("Bot: Thank you for using DIU Admission Chatbot.")
        break
    if user_question.strip() == "":
        print("Bot: Please enter a question.")
        continue
    processed_question = preprocess(user_question)
    user_vector = vectorizer.transform([processed_question])
    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )
    best_match = similarity.argmax()
    score = similarity[0][best_match]
    if score < 0.20:
        print(
            "Bot: Sorry, I could not find a suitable answer."
        )
        print(
            "Bot: Please contact DIU Admission Office "
            "for the latest information."
        )
    else:
        print(
            "Bot:",
            faqs[best_match]["answer"]
        )