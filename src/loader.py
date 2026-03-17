import os
import pypdf


def load_pdf(path):
    reader = pypdf.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page": i + 1,
                "text": text.strip(),
                "source": os.path.basename(path)
            })
    return pages


def load_all_pdfs(folder):
    all_pages = []
    if not os.path.exists(folder):
        os.makedirs(folder)
    for fname in os.listdir(folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(folder, fname)
            pages = load_pdf(path)
            all_pages.extend(pages)
    return all_pages
