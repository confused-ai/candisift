"""Text-extraction adapter: PDF / DOCX / image / plain text -> string.

Implements the TextExtractor port. Strategy per type:

- **PDF**: read the embedded text layer (pdfplumber). If a page has little/no
  text (scanned image PDF), render that page and OCR it with Tesseract. Pages
  that already have a text layer are kept verbatim — OCR runs only where needed.
- **Image** (png/jpg/tiff/…): OCR directly with Tesseract.
- **DOCX**: read paragraphs (python-docx). Embedded images are not OCR'd.
- **Everything else**: decode as UTF-8 (lossy).

Every optional dependency (pdfplumber, python-docx, pytesseract, pdf2image,
Pillow) and the external `tesseract`/`poppler` binaries are imported/called
lazily inside try/except. A missing dep or a corrupt file degrades to the best
text we already have (or "") and logs a warning — ingestion never crashes.

System deps for OCR: `tesseract` (the OCR engine) and `poppler` (PDF->image,
used by pdf2image). macOS: `brew install tesseract poppler`. Debian/Ubuntu:
`apt-get install tesseract-ocr poppler-utils`.
"""
from __future__ import annotations

import io
import logging
import os

log = logging.getLogger("candisift.parsing")

# extensions handled as raster images (OCR-only)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".ppm", ".pnm"}

# a page/image yielding fewer than this many non-space chars is treated as
# "no real text layer" -> OCR it. Tuned to skip headers/page numbers only.
_OCR_MIN_CHARS = 24

# decompression-bomb guard: refuse to rasterize an image above this many pixels.
# A 300-DPI A4 page is ~8.7MP; 40MP leaves headroom while blocking a crafted file
# that expands to gigapixels and exhausts memory. Pillow raises on exceeding it.
_MAX_IMAGE_PIXELS = 40_000_000


class FileTextExtractor:
    """Filename-dispatched extractor with an optional Tesseract OCR fallback.

    Args:
        ocr: master switch. When False, behaves as a pure text-layer extractor.
        ocr_lang: Tesseract language(s), e.g. "eng" or "eng+deu".
        ocr_dpi: render resolution for PDF pages. 300 is the OCR sweet spot;
            higher is slower with little accuracy gain on resumes.
        max_ocr_pages: hard cap on pages rendered+OCR'd per PDF (memory/time
            guard against a 500-page scan). Pages beyond it keep their text
            layer if any, else are skipped.
    """

    def __init__(
        self,
        ocr: bool = True,
        ocr_lang: str = "eng",
        ocr_dpi: int = 300,
        max_ocr_pages: int = 50,
        ocr_timeout_s: int = 30,
    ) -> None:
        self.ocr = ocr
        self.ocr_lang = ocr_lang
        self.ocr_dpi = max(72, int(ocr_dpi))
        self.max_ocr_pages = max(1, int(max_ocr_pages))
        # per-call wall-clock cap on the tesseract/poppler subprocesses so a
        # hostile or pathological file can't wedge a worker thread indefinitely.
        self.ocr_timeout_s = max(1, int(ocr_timeout_s))

    # ---- port entrypoint -------------------------------------------------
    def extract(self, content: bytes, filename: str) -> str:
        if not content:
            return ""
        ext = os.path.splitext((filename or "").lower())[1]
        if ext == ".pdf":
            return self._pdf(content)
        if ext == ".docx":
            return self._docx(content)
        if ext in IMAGE_EXTS:
            return self._image_ocr(content)
        # .txt/.md/.csv/unknown — decode lossily, never raise
        return content.decode("utf-8", errors="ignore").strip()

    # ---- PDF -------------------------------------------------------------
    def _pdf(self, content: bytes) -> str:
        text, needs_ocr = self._pdf_pymupdf4llm(content)
        if text and not needs_ocr:
            return text

        # Fallback to pdfplumber text layer
        pages = self._pdf_text_layer(content)
        text_plumber = "\n".join(pages).strip()

        if not self.ocr:
            return text or text_plumber

        # OCR only if the doc looks scanned: no text layer at all, or some page
        # is near-empty (mixed digital + scanned PDFs are common).
        needs_ocr = needs_ocr or (not pages) or any(len(p.strip()) < _OCR_MIN_CHARS for p in pages)
        if not needs_ocr:
            return text or text_plumber

        ocr_text = self._pdf_ocr(content, pages)
        # prefer the merged OCR result; fall back to whatever text layer existed
        return ocr_text or text or text_plumber

    def _pdf_pymupdf4llm(self, content: bytes) -> tuple[str, bool]:
        """Try pymupdf4llm to extract Markdown. Returns (text, needs_ocr)."""
        try:
            import fitz
            import pymupdf4llm
        except Exception:
            log.warning("pymupdf4llm not installed — cannot extract Markdown from PDF")
            return "", True

        doc = None
        try:
            doc = fitz.Document(stream=content, filetype="pdf")
            text = pymupdf4llm.to_markdown(doc)
            # check if it needs OCR (basically empty)
            needs_ocr = len(text.strip()) < _OCR_MIN_CHARS * len(doc)
            return text, needs_ocr
        except Exception as e:
            log.warning("pymupdf4llm read failed (%s)", e.__class__.__name__)
            return "", True
        finally:
            if doc is not None:
                doc.close()

    @staticmethod
    def _pdf_text_layer(content: bytes) -> list[str]:
        """Per-page text layer. [] if pdfplumber missing or PDF unreadable."""
        try:
            import pdfplumber  # optional dep
        except Exception:
            log.warning("pdfplumber not installed — cannot read PDF text layer")
            return []
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return [(page.extract_text() or "") for page in pdf.pages]
        except Exception as e:  # encrypted / corrupt / not-a-pdf
            log.warning("PDF text-layer read failed (%s); will try OCR", e.__class__.__name__)
            return []

    def _pdf_ocr(self, content: bytes, pages: list[str]) -> str:
        """Render up to max_ocr_pages and OCR pages lacking a text layer.

        Pages that already have text are kept as-is (no needless OCR). Returns
        "" if OCR deps/binaries are unavailable so the caller can fall back.
        """
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            from PIL import Image
        except Exception:
            log.warning("OCR skipped: install `pytesseract` and `pdf2image` for scanned PDFs")
            return ""
        Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS  # decompression-bomb guard

        # how many pages to consider: the text-layer count if we have it, else the cap.
        # Rendering ALL pages at once (the old convert_from_bytes(last_page=cap)) held
        # every page bitmap in memory simultaneously — ~1.3GB for 50 pages @300DPI.
        # Convert ONE page at a time so peak memory is a single page bitmap.
        limit = min(len(pages), self.max_ocr_pages) if pages else self.max_ocr_pages
        out: list[str] = []
        rendered = 0
        for i in range(1, limit + 1):
            existing = pages[i - 1].strip() if (i - 1) < len(pages) else ""
            if len(existing) >= _OCR_MIN_CHARS:
                out.append(existing)        # has a text layer -> no need to rasterize
                continue
            try:
                imgs = convert_from_bytes(
                    content, dpi=self.ocr_dpi, first_page=i, last_page=i,
                    timeout=self.ocr_timeout_s,
                )
            except Exception as e:
                log.warning("PDF->image failed on page %d (%s); is `poppler` installed?",
                            i, e.__class__.__name__)
                break
            if not imgs:                    # past the last page (unknown-length scan)
                break
            rendered = i
            image = imgs[0]
            try:
                out.append(self._ocr_image(pytesseract, image) or existing)
            finally:
                image.close()               # free the page bitmap before the next one

        merged = "\n".join(p for p in out if p).strip()
        # carry over any text-layer pages beyond what we rendered so nothing is dropped
        if len(pages) > rendered:
            tail = "\n".join(p.strip() for p in pages[rendered:] if p.strip())
            merged = (merged + "\n" + tail).strip() if tail else merged
        return merged

    # ---- image -----------------------------------------------------------
    def _image_ocr(self, content: bytes) -> str:
        if not self.ocr:
            return ""
        try:
            import pytesseract
            from PIL import Image
        except Exception:
            log.warning("OCR skipped: install `pytesseract` and `Pillow` for image files")
            return ""
        Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS  # decompression-bomb guard
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                return self._ocr_image(pytesseract, image)
        except Exception as e:
            # includes Pillow's DecompressionBombError on oversized images
            log.warning("image OCR failed (%s)", e.__class__.__name__)
            return ""

    def _ocr_image(self, pytesseract, image) -> str:
        """One Tesseract call. Returns "" on any engine/binary error."""
        try:
            return pytesseract.image_to_string(
                image, lang=self.ocr_lang, timeout=self.ocr_timeout_s
            ).strip()
        except Exception as e:
            # TesseractNotFoundError, missing lang data, etc.
            log.warning("tesseract OCR error (%s); check tesseract binary + langpack", e.__class__.__name__)
            return ""

    # ---- DOCX ------------------------------------------------------------
    def _docx(self, content: bytes) -> str:
        try:
            import docx  # python-docx, optional dep
        except Exception:
            log.warning("python-docx not installed — cannot read .docx")
            return ""
        try:
            document = docx.Document(io.BytesIO(content))
        except Exception as e:
            log.warning("DOCX read failed (%s)", e.__class__.__name__)
            return ""
        parts = [p.text for p in document.paragraphs]
        # include table cells — resumes often put skills/dates in tables
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        parts.append(cell.text)
        text = "\n".join(t for t in parts if t).strip()
        if self.ocr:
            text = self._docx_image_ocr(document, text)
        return text.strip()

    def _docx_image_ocr(self, document, text: str) -> str:
        """OCR images embedded in a DOCX (some resumes are a single image in a
        Word wrapper). Appends recovered text; logos/icons just yield little.
        Returns `text` unchanged if OCR deps are missing."""
        try:
            import pytesseract
            from PIL import Image
        except Exception:
            return text
        Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS  # decompression-bomb guard
        chunks = [text] if text else []
        seen = 0
        for rel in document.part.rels.values():
            if "image" not in rel.reltype or rel.is_external:
                continue
            if seen >= self.max_ocr_pages:
                break
            seen += 1
            try:
                blob = rel.target_part.blob
                with Image.open(io.BytesIO(blob)) as image:
                    image.load()
                    recovered = self._ocr_image(pytesseract, image)
            except Exception as e:
                log.warning("docx image OCR failed (%s)", e.__class__.__name__)
                continue
            if recovered:
                chunks.append(recovered)
        return "\n".join(chunks)


if __name__ == "__main__":  # ponytail: smallest real check of the OCR path
    import sys

    x = FileTextExtractor(ocr_dpi=200)

    # plain text path
    assert x.extract(b"hello world", "a.txt") == "hello world"
    # empty / unknown bytes never raise
    assert x.extract(b"", "a.pdf") == ""

    # OCR round-trip: render text to a PNG, then read it back via Tesseract.
    try:
        from PIL import Image, ImageDraw, ImageFont
        import pytesseract  # noqa: F401  (ensures binary+lib present before asserting)
    except Exception:
        print("SKIP OCR check: Pillow/pytesseract not installed")
        sys.exit(0)
    try:
        font = ImageFont.load_default(size=48)  # Pillow >=10
    except Exception:
        font = ImageFont.load_default()
    img = Image.new("RGB", (700, 160), "white")
    ImageDraw.Draw(img).text((20, 50), "Resume OCR works", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    got = x.extract(buf.getvalue(), "scan.png").lower()
    # large clean render -> Tesseract should recover at least one full word
    assert any(w in got for w in ("resume", "works", "ocr")), f"OCR returned: {got!r}"
    print("OK: text, empty, and OCR paths verified")
