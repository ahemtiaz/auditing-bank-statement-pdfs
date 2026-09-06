import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
import pdfplumber

class BasePdfParser(ABC):
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.total_parse_time = 0.0
        self.files_processed = 0

    @abstractmethod
    def partition_pdf(self, file_path: Path):
        pass

    def _save_text(self, text_data: list, file_path: Path, lib_name: str) -> None:
        file_stem = file_path.stem
        result_path = self.results_dir / lib_name
        Path(result_path).mkdir(parents=True, exist_ok=True)
        result_file = result_path / f"{file_stem}_{lib_name}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(text_data, f, ensure_ascii=False, indent=4)

    def _get_difficulty(self, file_path: Path) -> str:
        """Extract difficulty level from file path (easy/medium/hard)."""
        path_parts = file_path.parts
        for part in path_parts:
            if part.lower() in ['easy', 'medium', 'hard']:
                return part.lower()
        return 'unknown'  # fallback if no difficulty found

    def _log_progress(self, file_path: Path, index: int, total: int) -> None:
        print(f"[{self.lib_name}] Processing {file_path.name} ({index}/{total})")

    def get_timing_summary(self) -> str:
        """Return a formatted string with timing information."""
        avg_time = self.total_parse_time / self.files_processed if self.files_processed > 0 else 0
        return (f"Parser: {self.lib_name} | "
                f"Total time: {self.total_parse_time:.2f}s | "
                f"Files: {self.files_processed} | "
                f"Avg per file: {avg_time:.3f}s")


class PdfPlumberParser(BasePdfParser):
    def __init__(self, results_dir: Path) -> None:
        super().__init__(results_dir)
        self.lib_name = 'pdfplumber'

    def partition_pdf(self, file_path: Path):
        start_time = time.time()
        try:
            stem = file_path.stem
            page_num = int(''.join(filter(str.isdigit, stem))[-2:] or '1')
            difficulty = self._get_difficulty(file_path)
            text_data = []
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages, start=1):
                    self._log_progress(file_path, i, total_pages)
                    content = page.extract_text() or ""
                    text_data.append({
                        'page_number': i,
                        'text': content.strip(),
                        'difficulty': difficulty
                    })
            self._save_text(text_data, file_path, self.lib_name)
        except Exception as e:
            print(f"Failed to parse {file_path} with {self.lib_name}: {e}")
        finally:
            parse_time = time.time() - start_time
            self.total_parse_time += parse_time
            self.files_processed += 1
