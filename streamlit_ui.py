
import streamlit as st
import os
from src.document_analyzer.data_ingestion import DocumentHandler
from src.document_analyzer.data_analysis import DocumentAnalyzer


st.title("Document Portal - AI Analysis")

tab1, tab2 = st.tabs(["Single Document Analysis", "Document Comparison"])

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="single")
    if uploaded_file:
        handler = DocumentHandler()
        file_path = handler.save_pdf(uploaded_file)
        st.success(f"Uploaded {uploaded_file.name}")
        text = handler.read_pdf(file_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        st.write("Analysis Result:", result)

with tab2:
    st.subheader("Compare Two PDF Documents")
    file1 = st.file_uploader("Upload first PDF", type=["pdf"], key="compare1")
    file2 = st.file_uploader("Upload second PDF", type=["pdf"], key="compare2")
    if file1 and file2:
        handler1 = DocumentHandler()
        handler2 = DocumentHandler()
        path1 = handler1.save_pdf(file1)
        path2 = handler2.save_pdf(file2)
        text1 = handler1.read_pdf(path1)
        text2 = handler2.read_pdf(path2)
        from src.document_compare.retrieval import compare_documents
        result = compare_documents(text1, text2)
        st.write(f"Similarity Ratio: {result['similarity']:.2f}")
        st.write("Differences:")
        st.code("\n".join(result['differences']) if result['differences'] else "No significant differences found.")

st.title("Document Portal - AI Analysis")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    file_path = os.path.join("data", "uploaded_" + uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")
    # Analyze document (dummy function for now)
    result = analyze_document(file_path) # type: ignore
    st.write("Analysis Result:", result)
