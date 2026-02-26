from .dependencies import active_session, embeddings,llm,Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from flashrank import Ranker
from typing import List,Dict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.retrievers.self_query.base import AttributeInfo
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.runnables import RunnableLambda
import uuid
import os

store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="Exact lesson topic",
        type="string"
    ),
    AttributeInfo(
        name="subtopic",
        description="Narrow concept (e.g., light-dependent reactions)",
        type="string"
    ),
    AttributeInfo(
        name="subject",
        description="Subject area",
        type="string"
    ),
    AttributeInfo(
        name="grade_level",
        description="Grade or age range",
        type="string"
    ),
    AttributeInfo(
        name="learning_objective",
        description="Student outcome for this topic",
        type="string"
    ),
    AttributeInfo(
        name="keywords",
        description="Important terms and concepts",
        type="string"
    ),
]



document_content_description = "Excerpts from user-uploaded PDF documents (textbooks, articles, reports, etc.)"


async def process_pdf(file_path: str,session_id:str):
    loader=PyPDFLoader(file_path)
    documents: List[Document] = loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks=text_splitter.split_documents(documents)

    bm25_retriever=BM25Retriever.from_documents(chunks)
    bm25_retriever.k=8

    bm25_retriever.preprocess_func = lambda text: str(text).split()

    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"temp_{session_id}"
    )

    
    vector_retrieval = vectorstore.as_retriever(
        search_type="mmr",         
        search_kwargs={"k": 8,"fetch_k": 20, "lambda_mult": 0.5}             
    )

    client=Ranker()
        
    compressor=FlashrankRerank(
        client=client,
        model="ms-marco-MiniLM-L-12-v2",
        top_n=6
    )

       

    qa_prompt = PromptTemplate.from_template(
        """You are an expert teacher with 15+ years of experience creating engaging, standards-aligned lesson plans.

        Your ONLY source of information is the provided textbook excerpts below.
        Do NOT use any external knowledge, assumptions, or made-up content.

        Task: Create a complete, ready-to-use lesson plan based exactly on the content in the provided context.

        User request: {input}

        Retrieved textbook excerpts (from the uploaded document):
        {context}

        Follow this step-by-step reasoning process before writing the final lesson plan:

        1. Read the user request carefully and identify the main topic/focus.
        2. Scan the provided context and list (in your mind) all relevant key ideas, terms, facts, experiments, diagrams, or exercises that relate to the request.
        3. Decide on a short, catchy title (8-12 words max) that captures both the user request and the core content from the document.
        4. Determine the most appropriate grade level and subject based ONLY on the context (default to High School Biology Class XI-XII if no clear indication).
        5. Define 3-5 clear, measurable learning objectives using Bloom's taxonomy verbs (remember, understand, apply, analyze, evaluate, create).
        6. For each required section below, check if the context contains enough information:
           - If yes → write concrete content based ONLY on the excerpts.
           - If not enough information → write exactly: "Insufficient information in the document for this section."
        7. Be concise, practical, and 100% faithful to the excerpts — never invent activities, materials, or explanations that are not supported by the text.

        Now write the lesson plan using this exact markdown structure:

        ## Lesson Title
        Make the title short (8-12 words max), catchy, and focused on the main user request + key topic.

        ## Grade Level & Subject
        [Inferred or default: High School Biology (Class XI-XII level)]

        ## Duration
        [e.g. 45=60 minutes]

        ## Learning Objectives
        - Bullet list of 3-5 objectives

        ## Key Concepts & Vocabulary
        - Bullet list of main ideas and terms

        ## Materials Needed
        - List of implied or directly mentioned resources

        ## Lesson Structure
        ### 1. Introduction / Hook (5-10 min)
        [Engaging starter activity or question and explain with an real life example]

        ### 2. Direct Instruction / Explanation (10-15 min)
        [How to present and explain or describe core content]

        ### 3. Guided Practice / Activities (15-0 min)
        - For guided activities, prefer hands-on or discussion-based ideas directly supported by or logically extended from the text (e.g., group calculations, diagram drawing, experiment walkthrough).

        ### 4. Independent Practice (if applicable)
        [Student work time ideas]

        ### 5. Closure / Assessment (5-10 min)
        - Exit ticket, quick quiz, or reflection
        - Simple formative assessment

        ## Differentiation
        - One idea for supporting struggling learners
        - One idea for extending advanced learners

        ## Homework / Extension (optional)
        Suggest 1-2 logical follow-up tasks or readings based only on concepts covered (e.g., review previous chapter, solve related exercise, simple research question). Prefix with "Suggested:" if not directly in text.

        ## Source References
        - List key pages/sections used (from metadata if available)

        Think step-by-step, then produce only the formatted lesson plan — no additional explanations.
    """
    )


    question_answer_chain=create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )


    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retrieval],
        weights=[0.5, 0.5]  
    )

    compressor_retriever=ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=compressor
    )

    rag_chain = create_retrieval_chain(
        compressor_retriever,
        question_answer_chain
    )
    

    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    active_session[session_id]={
        "vectorstore":vectorstore,
        "chain":conversational_rag_chain,
        "filename":os.path.basename(file_path)
    }

    print(f"Session stored: {session_id} | Active sessions now: {list(active_session.keys())}")

    return len(documents),len(chunks)