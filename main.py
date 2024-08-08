import streamlit as st
import pdfplumber
import tempfile
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
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
        model_name="gpt-4o-mini",
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
    st.set_page_config(
        page_title="제조가부 AI - Spec 요약",
        page_icon="🔧",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("📄 제조가부 AI - Spec 요약 및 비교")
    st.markdown(
        """
        **환영합니다!**  
        이 앱은 PDF 형식의 수요가 사양(spec)을 업로드하고, 제조에 필요한 중요한 내용을 요약해드립니다.  
        또한 표준 사양과의 비교를 통해 차이점을 분석합니다.
        """
    )

    st.sidebar.header("⚙️ 설정")
    with st.sidebar.expander("ℹ️ 정보"):
        st.markdown("이 앱은 PDF 문서에서 텍스트를 추출하고 GPT-4 모델을 사용하여 내용을 요약합니다.")

    uploaded_file = st.file_uploader("🔄 수요가 Spec 업로드", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("PDF 파일 처리 중..."):
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(uploaded_file.getbuffer())
                pdf_path = temp_pdf.name

            # PDF 내용 추출 및 요약
            pdf_tool = PDFPlumberTool(pdf_path)
            extracted_text = pdf_tool.extract_text()
            summarized_text = summarize_pdf_content(extracted_text)

        st.success("요약 작업이 완료되었습니다!")
        
        # 요약된 내용 화면에 표시
        st.subheader("📋 요약된 내용:")
        st.code(summarized_text, language="markdown")

        # 요약된 내용 저장 및 다운로드 버튼 추가
        st.download_button(
            label="💾 요약된 내용 다운로드 (Text)",
            data=summarized_text,
            file_name="summarized_spec.txt",
            mime="text/plain"
        )

    else:
        st.info("먼저 PDF 파일을 업로드해주세요.")

if __name__ == "__main__":
    main()
