import streamlit as st
import pdfplumber
import tempfile
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from crewai import Crew, Agent, Task


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

# AGENT 정의
class Agents:
    def document_comparator(self, summarized_text, comparison_text):
        return Agent(
            role="Document Comparator",
            goal="Compare the summarized content from the uploaded document with another standard document to identify key considerations, discrepancies, and necessary adjustments.",
            tools=[],  # 도구 사용 없이 직접 작업
            allow_delegation=False,
            verbose=True,
            backstory="""
            You specialize in comparing client-specific documentation with industry standards.
            Your role is to identify any discrepancies, necessary adjustments, or key considerations between the uploaded document and the selected standard.
            """,
            context={'summarized_text': summarized_text, 'comparison_text': comparison_text}  # 컨텍스트로 요약된 텍스트를 전달
        )

# 작업 정의
class Tasks:
    def compare_with_selected_standard(self, agent):
        return Task(
            description="Compare the summarized content from the uploaded document with the selected standard to identify necessary adjustments, discrepancies, and considerations.",
            expected_output="A detailed comparison report that identifies key considerations, discrepancies, and adjustments needed when aligning the uploaded document with the selected standard.",
            agent=agent,
            output_file="comparison_report.md",
        )

    def final_summary(self, agent, context):
        return Task(
            description="Compile the comparison report and other relevant findings into a final summary document.",
            expected_output="A comprehensive final report that includes the comparison with the selected standard and summarizes necessary adjustments and key considerations.",
            agent=agent,
            context=context,
            output_file="final_summary_with_comparison.md",
        )


# Streamlit 웹 애플리케이션
def main():
    st.title("제조가부 AI")

    # 탭 설정
    tab1, tab2 = st.tabs(["수요가 Spec 요약", "표준 Spec 비교"])

    with tab1:
        st.header("수요가 Spec 요약")
        uploaded_file = st.file_uploader("수요가 Spec(pdf)", type="pdf", key="tab1_file_uploader")

        if uploaded_file is not None:
            # 파일을 이미 처리한 경우 다시 처리하지 않도록 방지
            if "summarized_text" not in st.session_state:
                with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                    temp_pdf.write(uploaded_file.getbuffer())
                    pdf_path = temp_pdf.name

                st.write("PDF 파일 업로드 완료. 작업을 시작합니다...")

                # PDF 내용 추출 및 요약
                pdf_tool = PDFPlumberTool(pdf_path)
                extracted_text = pdf_tool.extract_text()
                summarized_text = summarize_pdf_content(extracted_text)

                # 요약된 내용 저장
                st.session_state["summarized_text"] = summarized_text

            st.write("요약된 내용:")
            st.text(st.session_state["summarized_text"])

            # 다운로드 버튼 추가
            st.download_button(
                label="요약된 내용 다운로드 (Text)",
                data=st.session_state["summarized_text"],
                file_name="summarized_spec.txt",
                mime="text/plain"
            )

    with tab2:
        st.header("표준 Spec 비교")
        uploaded_file = st.file_uploader("수요가 Spec(pdf)", type="pdf", key="tab2_file_uploader_spec")
        comparison_file = st.file_uploader("표준 Spec(pdf)", type="pdf", key="tab2_file_uploader_comparison")

        if uploaded_file is not None and comparison_file is not None:
            # 파일을 이미 처리한 경우 다시 처리하지 않도록 방지
            if "final_summary" not in st.session_state:
                with tempfile.NamedTemporaryFile(delete=False) as temp_pdf1, tempfile.NamedTemporaryFile(delete=False) as temp_pdf2:
                    temp_pdf1.write(uploaded_file.getbuffer())
                    pdf_path = temp_pdf1.name

                    temp_pdf2.write(comparison_file.getbuffer())
                    comparison_path = temp_pdf2.name

                st.write("PDF 파일 업로드 완료. 표준 문서와의 비교 작업을 시작합니다...")

                # PDF 내용 추출 및 요약
                pdf_tool = PDFPlumberTool(pdf_path)
                comparison_tool = PDFPlumberTool(comparison_path)

                extracted_text = pdf_tool.extract_text()
                comparison_text = comparison_tool.extract_text()
                summarized_text = summarize_pdf_content(extracted_text)

                # AGENTS 및 TASKS 초기화
                agents = Agents()
                comparator_agent = agents.document_comparator(summarized_text, comparison_text)

                # 작업 생성
                tasks = Tasks()
                compare_task = tasks.compare_with_selected_standard(comparator_agent)
                final_summary_task = tasks.final_summary(comparator_agent, [compare_task])

                # Crew 초기화
                crew = Crew(
                    agents=[comparator_agent],
                    tasks=[compare_task, final_summary_task],
                    verbose=2,
                )

                # 작업 수행
                crew.kickoff()

                # 최종 보고서 읽기
                with open("final_summary_with_comparison.md", "r", encoding="utf-8") as file:
                    final_summary = file.read()

                st.session_state["final_summary"] = final_summary

            st.write("최종 보고서:")
            st.text(st.session_state["final_summary"])

            # 다운로드 버튼 추가
            st.download_button(
                label="최종 보고서 다운로드 (Text)",
                data=st.session_state["final_summary"],
                file_name="final_summary_with_comparison.txt",
                mime="text/plain"
            )

    st.success("작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
