"""Narrative text extraction for RAG.

Sectioning and chunking logic is carried over from the original edgar_parser.py.
Chunks record the filing accession. `python -m edgar_rag rebuild-chunks` re-extracts
(and re-embeds) filings after a change here.
"""

import re

# Cap per heading-delimited section. It guards against parse noise, but a filing whose whole
# risk-factor section sits under one heading (Meta) runs to ~300 chunks, and at the previous
# cap of 120 the rest was silently dropped.
MAX_CHUNKS_PER_SECTION = 500
# Below this share of the document's non-table text, the heading-based split is assumed to
# have missed the body (few usable headings), and the TOC / Item parsers are tried as well.
MIN_HEADED_COVERAGE = 0.2

SECTION_CANONICAL_MAP = {
    'risks related to our product offerings': 'Risk Factors - Product Offerings',
    'sensitivity analysis': 'Sensitivity Analysis',
    'industry trends and opportunities': 'Industry Trends and Opportunities',
    'commitments and contingencies': 'Commitments and Contingencies',
    'legal proceedings': 'Legal Proceedings',
    'note about forward-looking statements': 'Forward-Looking Statements',
}

SECTION_PATTERN_RULES = [
    ('incorporated herein by reference', 'Incorporation by Reference'),
    ('the information required by this item', 'Incorporation by Reference'),
    ('in addition, our business may be adversely affected if', 'Risk Factors - Additional Risks'),
    ('our international operations expose us to a number of risks', 'Risk Factors - International Operations'),
    ('risks relating to the evolution of our business', 'Risk Factors - Business Evolution'),
    ('operational risks', 'Risk Factors - Operational Risks'),
    ('legal and regulatory risks', 'Risk Factors - Legal and Regulatory'),
    ('strategic and competitive risks', 'Risk Factors - Strategic and Competitive'),
    ('general risks', 'Risk Factors - General'),
    ('critical accounting estimates', 'Critical Accounting Estimates'),
    ('liquidity and capital resources', 'Liquidity and Capital Resources'),
    ('business overview', 'Overview'),
    ('competition among platform-based ecosystems', 'Competition'),
]


def is_narrative_section(section_name):
    """Check if section contains narrative content vs. financial tables."""
    name_lower = section_name.lower()

    exclude_keywords = [
        'balance sheet', 'statement of financial position',
        'income statement', 'statement of operations', 'statement of earnings',
        'cash flow', 'statement of cash flows',
        'stockholders equity', 'shareholders equity', 'statement of equity',
        'statement of comprehensive income',
        'financial statements', 'consolidated statements',
        'table of contents', 'index to', 'index of'
    ]

    if any(kw in name_lower for kw in exclude_keywords):
        return False

    if any(kw in name_lower for kw in ['cover page', 'signatures', 'exhibits', 'header']):
        return False

    return True


def normalize_section_name(section_name):
    """Normalize verbose/sentence-like headings into stable section buckets."""
    if not section_name:
        return 'UNKNOWN'

    raw = re.sub(r'\s+', ' ', section_name.strip())
    lower = raw.lower()

    if lower in SECTION_CANONICAL_MAP:
        return SECTION_CANONICAL_MAP[lower]

    for key, canonical in SECTION_CANONICAL_MAP.items():
        if key in lower:
            return canonical

    for pattern, canonical in SECTION_PATTERN_RULES:
        if pattern in lower:
            return canonical

    if lower.startswith('highlights from the '):
        return 'Highlights'

    if lower.startswith('risks related to '):
        tail = raw[len('Risks Related To '):].strip(' :.-')
        return f"Risk Factors - {tail}" if tail else 'Risk Factors'

    if lower.startswith('note ') and ':' in raw:
        note_title = raw.split(':', 1)[0].strip()
        return note_title

    if len(raw) > 120 or raw.startswith(('•', '-', '*')):
        if 'risk' in lower:
            return 'Risk Factors'
        if 'highlight' in lower:
            return 'Highlights'
        if 'competition' in lower:
            return 'Competition'
        if 'sensitivity' in lower:
            return 'Sensitivity Analysis'
        if 'commitment' in lower or 'contingenc' in lower:
            return 'Commitments and Contingencies'
        return raw[:80].rstrip(' .:;-')

    if raw.isupper() and any(c.isalpha() for c in raw):
        return raw.title()

    return raw


def split_into_sections(markdown_text):
    """Split markdown document into sections based on headers."""
    sections = []
    current_section = {'name': 'HEADER', 'content': []}

    def is_noisy_header(header_text):
        candidate = header_text.strip()
        if not candidate:
            return True
        if candidate.startswith(('•', '-', '*')):
            return True
        if len(candidate) > 180:
            return True
        if sum(ch.isalpha() for ch in candidate) < 3:
            return True
        return False

    for line in markdown_text.split('\n'):
        header_match = re.match(r'^#{1,4}\s+(.+)$', line)
        if header_match:
            header_text = header_match.group(1).strip()

            if is_noisy_header(header_text):
                current_section['content'].append(line)
                continue

            if current_section['content']:
                sections.append({
                    'name': current_section['name'],
                    'content': '\n'.join(current_section['content'])
                })

            current_section = {'name': header_text, 'content': [line]}
        else:
            current_section['content'].append(line)

    if current_section['content']:
        sections.append({
            'name': current_section['name'],
            'content': '\n'.join(current_section['content'])
        })

    return sections


def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks for embedding."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para)

        if current_length + para_length > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = current_chunk[-1:] if overlap > 0 else []
            current_length = len(current_chunk[0]) if current_chunk else 0

        current_chunk.append(para)
        current_length += para_length

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


# ── fallback for documents without markdown headings ─────────
# Prospectuses (424B4) are often PDF-style HTML with absolutely positioned text, which
# converts to markdown with no headings and page-sized paragraphs. Their table of
# contents survives as rows like "| RISK FACTORS........ | 27 |", and each section in the
# body starts with "<page> <TITLE>", so sections are recovered from the TOC.

TOC_ROW = re.compile(r'^\|\s*(?P<title>[^|]*?)\s*\.{3,}[\s…]*\|\s*(?P<page>[ivxlcdm]+|\d{1,3}|F-\d+)\s*\|\s*$')
TOC_MAX_CHUNKS_PER_SECTION = 1000  # sections here are real document sections, not parse noise


def toc_sections(markdown_text):
    entries, toc_end, pos = [], 0, 0
    for line in markdown_text.split('\n'):
        m = TOC_ROW.match(line.strip())
        if m:
            title = m['title'].strip()
            if title.upper() == title and sum(c.isalpha() for c in title) >= 3 \
                    and title not in {t for t, _ in entries}:
                entries.append((title, m['page']))
                toc_end = pos + len(line)
        pos += len(line) + 1
    if len(entries) < 3:
        return []

    starts, cursor = [], toc_end
    for title, page in entries:
        # The TOC sometimes drops spaces ("RESULTSOF"), so match title characters with optional whitespace.
        title_re = r'\s*'.join(re.escape(c) for c in title if not c.isspace())
        m = (re.compile(rf'(?<![\w.]){re.escape(page)}\s+{title_re}').search(markdown_text, cursor)
             or re.compile(rf'(?<!\w){title_re}(?!\w)').search(markdown_text, cursor))
        if m:
            # Name from the body text (properly spaced), without the leading page number.
            heading = re.sub(rf'^{re.escape(page)}\s+', '', m.group()) if m.group().startswith(page) else m.group()
            starts.append((m.start(), m.end(), _heading_case(heading)))
            cursor = m.end()
    return [
        {'name': title, 'content': markdown_text[body_start:(starts[i + 1][0] if i + 1 < len(starts) else len(markdown_text))]}
        for i, (_, body_start, title) in enumerate(starts)
    ]


_SMALL_WORDS = {'a', 'an', 'and', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'with'}


def _heading_case(heading):
    words = re.sub(r'\s+', ' ', heading).strip().lower().split(' ')
    titled = [w if (i and w in _SMALL_WORDS) else w[:1].upper() + w[1:] for i, w in enumerate(words)]
    return ' '.join(titled).replace('u.s.', 'U.S.').replace('Non-u.s.', 'Non-U.S.')


def _pieces(text, max_len):
    """Sentence groups of at most ~max_len chars; word-boundary split for run-on text (lists, tables)."""
    current = ''
    for sentence in re.split(r'(?<=[.!?”"])\s+(?=[A-Z“"(])', text):
        while len(sentence) > 2 * max_len:
            cut = sentence.rfind(' ', 0, max_len)
            cut = cut if cut > 0 else max_len
            if current:
                yield current
                current = ''
            yield sentence[:cut]
            sentence = sentence[cut:].lstrip()
        if current and len(current) + len(sentence) + 1 > max_len:
            yield current
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        yield current


def split_long_paragraphs(text, max_len=600):
    """Break page-sized paragraphs into sentence groups so chunk_text can size chunks."""
    out = []
    for para in text.split('\n\n'):
        para = para.strip()
        out.extend([para] if len(para) <= max_len else _pieces(para, max_len))
    return '\n\n'.join(out)


# ── fallback for 10-K/10-Q markdown without headings ─────────
# Some periodic filings (e.g. Dominion's 10-Ks, Oracle 2023-2025) also convert to markdown
# with no headings, and their TOCs don't use dot leaders. edgartools' structured parser
# (filing.obj()) still finds the standard Items, so narrative Items are taken from there.

ITEM_NAMES = {
    "10-K": {"Item 1": "Business", "Item 1A": "Risk Factors", "Item 1C": "Cybersecurity",
             "Item 2": "Properties", "Item 3": "Legal Proceedings",
             "Item 7": "Management’s Discussion and Analysis of Financial Condition and Results of Operations",
             "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk"},
    "10-Q": {"Part I, Item 2": "Management’s Discussion and Analysis of Financial Condition and Results of Operations",
             "Part I, Item 3": "Quantitative and Qualitative Disclosures About Market Risk",
             "Part II, Item 1": "Legal Proceedings", "Part II, Item 1A": "Risk Factors",
             "Part II, Item 5": "Other Information"},
}


def item_sections(filing):
    try:
        doc = filing.obj()
        keys = list(doc.items)
    except Exception as e:
        print(f"      ✗ Item parser failed: {e}")
        return []
    names = ITEM_NAMES.get(filing.form, {})
    sections = []
    for key in keys:
        # Some 10-Qs label items without the "Part I," prefix; those are Part I items.
        lookup = key if key in names or filing.form != "10-Q" or key.startswith("Part") else f"Part I, {key}"
        if lookup not in names:
            continue
        try:
            body = doc[key] or ""
        except Exception:
            continue
        sections.append({"name": names[lookup], "content": _clean_item_text(body)})
    return sections


def _clean_item_text(body):
    """Drop flattened table cells: numeric fragments and short non-heading lines."""
    keep = []
    for para in body.split("\n\n"):
        para = para.strip()
        letters = sum(c.isalpha() for c in para)
        if not para or letters < 0.5 * len(para):
            continue
        if len(para) < 40 and not para.isupper():
            continue
        keep.append(para)
    return "\n\n".join(keep)


def extract_narrative_chunks(filing, ticker):
    """Extract narrative text chunks from the filing's markdown. Returns (chunks, status)."""
    filing_date, filing_type, accession = filing.filing_date, filing.form, filing.accession_number
    try:
        markdown = filing.markdown()
    except Exception as e:
        print(f"      ✗ Could not get markdown: {e}")
        return [], "failed"

    # Headed markdown (the common case) first. If that yields nothing -- markdown with no
    # usable headings, where everything lands in a 'HEADER' section that is filtered out --
    # recover real sections from a prospectus-style TOC or the 10-K/10-Q Item parser.
    # The fallback also runs when the headings only caught a sliver of the text (Dominion's
    # 2023-24 10-Qs have headings for the short Items but none for MD&A: 5K of 218K characters),
    # and whichever split recovers more text wins.
    pieces = _chunk_sections(split_into_sections(markdown), MAX_CHUNKS_PER_SECTION, split_paragraphs=False)
    body_chars = sum(len(line) for line in markdown.split('\n') if '|' not in line)
    if _chars(pieces) < MIN_HEADED_COVERAGE * body_chars:
        sections = toc_sections(markdown) or (item_sections(filing) if filing_type in ("10-K", "10-Q") else [])
        recovered = _chunk_sections(sections, TOC_MAX_CHUNKS_PER_SECTION, split_paragraphs=True)
        if _chars(recovered) > _chars(pieces):
            pieces = recovered

    return [{
        'accession': accession,
        'ticker': ticker,
        'filing_date': filing_date,
        'filing_type': filing_type,
        'section': section_name,
        'chunk_index': i,
        'chunk_text': chunk,
    } for section_name, i, chunk in pieces], "ok"


def _chars(pieces):
    return sum(len(chunk) for _, _, chunk in pieces)


def _chunk_sections(sections, max_chunks, split_paragraphs):
    """[(section_name, chunk_index, chunk_text)] for the narrative sections."""
    out = []
    next_index = {}  # sections can repeat a name (e.g. "Revenue" under quarter and YTD discussions)
    for section in sections:
        section_name = normalize_section_name(section['name'])
        section_content = section['content']

        if not is_narrative_section(section_name):
            continue

        if len(section_content) < 200:
            continue

        lines = section_content.split('\n')
        narrative_lines = []
        in_table = False

        for line in lines:
            if '|' in line:
                in_table = True
            elif in_table:
                in_table = False
                narrative_lines.append('')

            if not in_table:
                narrative_lines.append(line)

        narrative = '\n'.join(narrative_lines).strip()

        if len(narrative) < 200:
            continue

        if split_paragraphs:
            narrative = split_long_paragraphs(narrative)
        chunks = chunk_text(narrative)
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

        # Continue numbering across same-named sections; restarting at 0 used to collide on
        # the (accession, section, chunk_index) key and silently drop the later section.
        start = next_index.get(section_name, 0)
        out.extend((section_name, start + i, chunk) for i, chunk in enumerate(chunks))
        next_index[section_name] = start + len(chunks)
    return out


CHUNK_COLUMNS = ["accession", "ticker", "filing_date", "filing_type", "section", "chunk_index", "chunk_text"]
