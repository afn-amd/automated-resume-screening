import streamlit as st
import pickle
from docx import Document
from PyPDF2 import PdfReader
import nltk
import string
import re
import time

nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# ======= UTIL FUNCTIONS ========

def cleanResume(txt):
    cleanText = re.sub(r'http\S+|#\S+|@\S+', '', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    punc_pattern = r'[%s]' % re.escape(string.punctuation)
    cleanText = re.sub(punc_pattern, ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    return ''.join(page.extract_text() for page in pdf_reader.pages)

def extract_text_from_docx(file):
    doc = Document(file)
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = clf.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# ====== STREAMLIT UI =========
def main():

    st.set_page_config(page_title="Resume Categorizer", page_icon="📄", layout="wide")

    # HEADER
    st.markdown(
        """
        <h1 style='text-align: center; color: #3a6ff7;'>📄 Resume Job Category Predictor</h1>
        <p style='text-align:center; font-size:18px;'>Upload a resume in PDF, DOCX, or TXT format and let AI predict the suitable job category</p>
        """,
        unsafe_allow_html=True
    )

    st.write("---")

    # FILE UPLOAD SECTION
    uploaded_file = st.file_uploader(
        "📁 Upload Resume",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT"
    )

    if uploaded_file is not None:
        with st.spinner("Extracting text from resume..."):
            resume_text = handle_file_upload(uploaded_file)
            time.sleep(1)

        st.success("Text extracted successfully!")

        if st.checkbox("Show extracted resume content"):
            st.text_area("📌 Extracted Content", resume_text, height=280)

        st.write("---")
        st.subheader("🔎 Predicting job category...")

        with st.spinner("Analyzing resume using AI model..."):
            time.sleep(1)
            category = pred(resume_text)

        st.balloons()
        st.success(f"🎯 **Predicted Job Category:** `{category}`")

        st.write("---")
        st.info("⚡ Tip: Try uploading different resumes to compare predictions!")

    else:
        st.warning("Please upload a resume to begin.")

    st.markdown(
        """
        <footer style='text-align:center; margin-top:40px; color:gray;'>
        ✨ Developed with ❤️ by Affan Ahmed
        </footer>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
