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