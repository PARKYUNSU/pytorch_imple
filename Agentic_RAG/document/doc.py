import os

class DocumentLoaderManager:
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs
        self.loader = self._select_loader()

    def _select_loader(self):
        # 웹 URL인 경우
        if self.file_path.startswith("http"):
            from langchain_community.document_loaders import WebBaseLoader
            return WebBaseLoader(web_paths=(self.file_path,), **self.kwargs)
        
        # 파일 확장자 기반 선택
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(self.file_path, **self.kwargs)
        elif ext == ".csv":
            from langchain_community.document_loaders.csv_loader import CSVLoader
            return CSVLoader(file_path=self.file_path, **self.kwargs)
        elif ext in [".xls", ".xlsx"]:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            return UnstructuredExcelLoader(self.file_path, **self.kwargs)
        elif ext == ".docx":
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            return UnstructuredWordDocumentLoader(self.file_path, **self.kwargs)
        elif ext == ".pptx":
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            return UnstructuredPowerPointLoader(self.file_path, **self.kwargs)
        elif ext == ".hwp":
            from langchain_teddynote.document_loaders import HWPLoader
            return HWPLoader(self.file_path, **self.kwargs)
        elif ext == ".txt":
            from langchain_community.document_loaders import TextLoader
            return TextLoader(self.file_path, **self.kwargs)
        elif ext == ".json":
            from langchain_community.document_loaders import JSONLoader
            return JSONLoader(file_path=self.file_path, **self.kwargs)
        else:
            raise ValueError(f"지원되지 않는 파일 확장자입니다: {ext}")

    def load_documents(self):
        return self.loader.load()

    def lazy_load_documents(self):
        if hasattr(self.loader, "lazy_load"):
            return self.loader.lazy_load()
        else:
            raise AttributeError("현재 로더는 lazy_load()를 지원하지 않습니다.")
        
    def split_text(self, splitter_type="character", **splitter_kwargs):
        """
        splitter_type: 사용할 분할기 유형
            - "character": CharacterTextSplitter (기본 구분자는 "\n\n")
            - "recursive": RecursiveCharacterTextSplitter (재귀적으로 분할)
            - "token": TokenTextSplitter (토큰 수 기준 분할)
            - "spacy": SpacyTextSplitter (spaCy tokenizer 사용)
            - "sentence_transformers": SentenceTransformersTokenTextSplitter
            - "huggingface": CharacterTextSplitter.from_huggingface_tokenizer 사용 (hf_tokenizer 필요 시 제공)
            - "semantic_chunker": SemanticChunker (OpenAIEmbeddings 기반)
            - "markdown_header": MarkdownHeaderTextSplitter (마크다운 헤더 기준 분할)
            - "html_header": HTMLHeaderTextSplitter (HTML 헤더 기준 분할)
            - "recursive_json": RecursiveJsonSplitter (JSON 데이터를 재귀적으로 분할)
        
        splitter_kwargs: 각 텍스트 분할기에 필요한 추가 매개변수
        """
        docs = self.load_documents()
        if not docs:
            raise ValueError("문서를 불러올 수 없습니다.")
        text = docs[0].page_content

        # 선택한 분할기 유형에 따라 텍스트 분할기 초기화
        if splitter_type == "character":
            from langchain_text_splitters import CharacterTextSplitter
            splitter = CharacterTextSplitter(**splitter_kwargs)
        elif splitter_type == "recursive":
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
        elif splitter_type == "token":
            from langchain_text_splitters import TokenTextSplitter
            splitter = TokenTextSplitter(**splitter_kwargs)
        elif splitter_type == "spacy":
            from langchain_text_splitters import SpacyTextSplitter
            splitter = SpacyTextSplitter(**splitter_kwargs)
        elif splitter_type == "sentence_transformers":
            from langchain_text_splitters import SentenceTransformersTokenTextSplitter
            splitter = SentenceTransformersTokenTextSplitter(**splitter_kwargs)
        elif splitter_type == "huggingface":
            from langchain_text_splitters import CharacterTextSplitter
            # hf_tokenizer를 splitter_kwargs에서 가져오거나 기본값 사용
            hf_tokenizer = splitter_kwargs.pop("hf_tokenizer", None)
            if hf_tokenizer is None:
                from transformers import GPT2TokenizerFast
                hf_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            splitter = CharacterTextSplitter.from_huggingface_tokenizer(hf_tokenizer, **splitter_kwargs)
        elif splitter_type == "semantic_chunker":
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai.embeddings import OpenAIEmbeddings
            splitter = SemanticChunker(OpenAIEmbeddings(), **splitter_kwargs)
        elif splitter_type == "markdown_header":
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            splitter = MarkdownHeaderTextSplitter(**splitter_kwargs)
        elif splitter_type == "html_header":
            from langchain_text_splitters import HTMLHeaderTextSplitter
            splitter = HTMLHeaderTextSplitter(**splitter_kwargs)
        elif splitter_type == "recursive_json":
            from langchain_text_splitters import RecursiveJsonSplitter
            splitter = RecursiveJsonSplitter(**splitter_kwargs)
        else:
            raise ValueError(f"지원되지 않는 텍스트 분할기 유형입니다: {splitter_type}")
        
        return splitter.create_documents([text])