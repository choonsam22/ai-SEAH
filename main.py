import streamlit as st
import pdfplumber
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


# PDFPlumberTool 정의
class PDFPlumberTool:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text() + "\n"
        return pdf_text

# LLM을 사용해 PDF 내용을 정리하는 함수
def summarize_pdf_content(pdf_text):
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.1,
        max_tokens=1000,
    )
 
    prompt = f"""
    The most important thing is to keep in mind that there should be no missing content in the file.
    The following is the content extracted from a PDF document. Please summarize the relevant sections for steel pipe manufacturing:

    {pdf_text}

    Please focus only on sections related to steel pipe manufacturing and exclude any unrelated sections (e.g., aluminum, gaskets, bolts, flanges). Summarize the content in a structured table format, highlighting key items and requirements.

    **Example format 1:**
    Section: [Section Number]  Aluminum
    - Piping tolerances: according to ANSI H35.2.
    - Packaging: ASTM B660 is not required.
    - Heat treatment: ASTM B918 is not required, Solution treatment is acceptable.

    **Example format 2:**
    Section: [Section Number]
    - [Content Title]: [Key details and requirements related to steel pipe manufacturing].

    **Example format 3:**
    
    | Section  | Content  | Summary  |
    |----------|----------|----------|
    | [Section Number] | [Content Title] | [Summary of the key points related to steel pipe manufacturing]. |
    | [Section Number] | [Content Title] | [Summary of the key points related to steel pipe manufacturing]. |

    Please replace the placeholders like [Section Number], [Content Title], and [Summary] with the actual content from the PDF. Focus exclusively on sections that are relevant to steel pipe manufacturing and present them in the table format provided.
    Please return the summarized content in a table format, including the section number, content title, and a summary of the key points related to steel pipe manufacturing.
    
    Aluminum, gaskets, bolts, flanges are not the things we need to review.
    We are a company that manufactures steel pipes.
    We need to summarize only the information for manufacturing steel pipes.
    """
    summary = llm.predict(prompt)
    return summary.strip()

# Streamlit 웹 애플리케이션
def main():
    st.title("제조가부 AI")
    
    uploaded_file = st.file_uploader("수요가 Spec 업로드", type="pdf")
    
    if uploaded_file is not None:
        pdf_path = uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("PDF 파일 업로드 완료. 요약 작업을 시작합니다...")

        # PDF 내용 추출 및 요약
        pdf_tool = PDFPlumberTool(pdf_path)
        extracted_text = pdf_tool.extract_text()
        summarized_text = summarize_pdf_content(extracted_text)

        # 요약된 내용 화면에 표시
        st.write("요약된 내용:")
        st.markdown(summarized_text)

        # 요약된 내용 저장 및 다운로드 버튼 추가
        with open("summarized_spec.txt", "w", encoding="utf-8") as file:
            file.write(summarized_text)
        
        st.download_button(
            label="요약된 내용 다운로드 (Text)",
            data=summarized_text,
            file_name="summarized_spec.txt",
            mime="text/plain"
        )

        st.success("요약 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
