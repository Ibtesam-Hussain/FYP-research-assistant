import pymupdf4llm
from pathlib import Path
from dataclasses import dataclass, field

# CRITICAL FIX: Disable the fragile ONNX-based deep learning visual parser.
# This forces PyMuPDF4LLM to use its lightning-fast native structural engine,
# preventing the onnxruntime tensor type crashes on complex PDFs.
pymupdf4llm.use_layout(False)

@dataclass
class PageContent:
    page_num: int
    markdown_text: str

@dataclass
class ParsedDocument:
    source_file: str
    pages: list[PageContent] = field(default_factory=list)
    full_content: str = ""

def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """
    Parse a multi-column PDF into structured Markdown per page using PyMuPDF4LLM.
    """
    pdf_path = Path(pdf_path)
    parsed = ParsedDocument(source_file=pdf_path.name)

    # Convert the PDF file into a structured dictionary per page
    pages_data = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    full_text_parts = []
    for page_dict in pages_data:
        # PyMuPDF4LLM may expose the page number either at the top-level
        # key "page" or under "metadata". Use a robust fallback to avoid
        # silently defaulting to 0 for every page.
        page_num = page_dict.get("page")
        if page_num is None:
            page_num = page_dict.get("metadata", {}).get("page_number")
        if page_num is None:
            page_num = page_dict.get("metadata", {}).get("page", 0)

        page_text = page_dict.get("text", "").strip()

        if not page_text:
            continue

        parsed.pages.append(PageContent(page_num=page_num, markdown_text=page_text))
        full_text_parts.append(page_text)

    parsed.full_content = "\n\n".join(full_text_parts)
    return parsed

def parse_all_pdfs(raw_dir: str | Path) -> list[ParsedDocument]:
    """Parse every PDF in a directory. Skips non-PDF files silently."""
    raw_dir = Path(raw_dir)
    results = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        print(f"Parsing: {pdf_path.name}")
        try:
            results.append(parse_pdf(pdf_path))
        except Exception as e:
            print(f"❌ Failed to parse {pdf_path.name}: {e}")
    return results

def display_sanity_check(doc: ParsedDocument):
    """Prints a clean structural view of the parsed Markdown layout."""
    print("\n" + "=" * 80)
    print(f" SANITY CHECK FOR: {doc.source_file} ")
    print("=" * 80)

    for page in doc.pages:
        print(f"\n--- [ PAGE {page.page_num} ] ---")
        lines = page.markdown_text.splitlines()
        
        # Display only the first few lines of each page to keep terminal uncluttered
        preview_lines = lines[:]
        for line in preview_lines:
            print(f"  {line}")
            
        # if len(lines) > 15:
        #     print(f"  ... [{len(lines) - 15} more lines parsed on this page] ...")

if __name__ == "__main__":
    import sys

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    docs = parse_all_pdfs(raw_dir)
    
    # Run the visualization on the first document found
    if docs:
        display_sanity_check(docs[0])
