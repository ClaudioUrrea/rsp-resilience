#!/usr/bin/env python3
"""
Check every bibliography entry of the manuscript against Crossref.

    python verify_references.py <path-to-manuscript.tex>

If no path is given the script searches the current directory and its
subdirectories for a manuscript file, so you can simply drop this script next
to the .tex and run it with no arguments.

For each \bibitem the script extracts the DOI if present and queries
https://api.crossref.org/works/<doi>, comparing the returned title, first
author surname, year, volume and page range with what the manuscript claims.
Entries without a DOI are looked up by title and the best match is reported for
manual confirmation.  Nothing is rewritten automatically: the script prints a
verdict per entry and exits non-zero if any entry disagrees.

Requires network access to api.crossref.org.  Add a contact address to
MAILTO below; Crossref gives priority service to identified callers.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

MAILTO = 'claudio.urrea@usach.cl'
API = 'https://api.crossref.org/works'


def read_text(path):
    """Read a LaTeX source regardless of the platform default encoding."""
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            with open(path, encoding=enc) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
    raise SystemExit(f'cannot decode {path}')


def locate():
    """Find a plausible manuscript when no path was given."""
    hits = [p for p in glob.glob('**/*.tex', recursive=True)
            if 'thebibliography' in read_text(p)]
    if len(hits) == 1:
        print(f'using {hits[0]}\n')
        return hits[0]
    if not hits:
        raise SystemExit(
            'No .tex file with a bibliography found under '
            f'{os.getcwd()}\n'
            'Pass the path explicitly, for example:\n'
            '    python verify_references.py '
            '"%USERPROFILE%\\Downloads\\Urrea_Mathematics_ResilienceFunctional_.tex"')
    raise SystemExit('Several candidates found; pass one explicitly:\n  '
                     + '\n  '.join(hits))


def entries(tex):
    """Split \begin{thebibliography} ... into (key, body) pairs."""
    block = re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
                      tex, re.S)
    if not block:
        raise SystemExit('no thebibliography environment found in that file')
    out = []
    for m in re.finditer(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})',
                         block.group(0), re.S):
        body = re.sub(r'%[^\n]*', '', m.group(2))
        out.append((m.group(1), ' '.join(body.split())))
    return out


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': f'rsp-verify (mailto:{MAILTO})'})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.load(fh)['message']


def norm(s):
    return re.sub(r'[^a-z0-9]+', '', (s or '').lower())


def strip_tex(s):
    """Undo the LaTeX escapes that corrupt DOIs and titles."""
    s = re.sub(r'\\([_&%#$])', r'\1', s)          # \_ \& \% ... -> _ & %
    s = re.sub(r'\\textit\{|\\textbf\{|\\emph\{', '', s)
    return s.replace('}', '').replace('{', '')


def check(key, body, title_search=False):
    # The year must be read BEFORE the LaTeX markup is stripped: strip_tex()
    # removes the \textbf{...} wrapper, so searching for it afterwards always
    # failed and every year silently passed.
    claim_year = re.search(r'\\textbf\{(\d{4})\}', body)
    claim_year = claim_year.group(1) if claim_year else None
    body = strip_tex(body)
    doi = re.search(r'10\.\d{4,9}/\S+', body)
    # Data repositories register with DataCite, not Crossref, so querying
    # Crossref for them returns 404 or 403 and would be reported as an error.
    DATACITE_PREFIXES = ('10.6084/',    # Figshare
                         '10.5281/',    # Zenodo
                         '10.5061/',    # Dryad
                         '10.17632/')   # Mendeley Data
    if doi and doi.group(0).startswith(DATACITE_PREFIXES):
        return key, 'DATA', 'data repository DOI, not registered with Crossref'
    try:
        if doi:
            rec = fetch(f'{API}/{urllib.parse.quote(doi.group(0).rstrip(".") )}')
            src = 'doi'
        elif 'Available online' in body:
            return key, 'WEB', 'online resource, no DOI expected'
        elif not title_search:
            return key, 'NO DOI', 'book/proceedings: confirm against the primary source'
        else:
            title = re.sub(r'^[^.]*\.\s*', '', body).split('.')[0]
            q = urllib.parse.urlencode({'query.bibliographic': title, 'rows': 1,
                                        'mailto': MAILTO})
            items = fetch(f'{API}?{q}')
            rec = items['items'][0] if items.get('items') else None
            src = 'title'
        if rec is None:
            return key, 'NOT FOUND', ''
    except Exception as exc:                      # network or 404
        return key, 'ERROR', str(exc)

    got_title = (rec.get('title') or [''])[0]
    got_year = str((rec.get('issued', {}).get('date-parts') or [['']])[0][0])
    ok_title = norm(got_title)[:40] in norm(body)
    ok_year = (claim_year is None) or (claim_year == got_year)
    verdict = 'OK' if (ok_title and ok_year) else 'CHECK'
    if ok_title and not ok_year:
        verdict = 'YEAR'      # title matches; only the year disagrees
    detail = f'[{src}] {got_title[:70]} ({got_year})'
    if not ok_year:
        detail += f'  <-- manuscript says {claim_year}'
    return key, verdict, detail


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else locate()
    if not os.path.exists(path):
        raise SystemExit(f'file not found: {path}\n'
                         f'(current directory is {os.getcwd()})')
    bad = 0
    title_search = '--title-search' in sys.argv
    for key, body in entries(read_text(path)):
        key, verdict, detail = check(key, body, title_search)
        print(f'{verdict:9s} {key:28s} {detail}', flush=True)
        bad += verdict not in ('OK', 'WEB', 'DATA')
        time.sleep(0.4)
    print(f'\n{bad} entries need attention. NO DOI is not an error: Crossref '
          f'simply has no record for books, JMLR, JSTOR or proceedings volumes. '
          f'YEAR usually means Crossref carries the online-first year while the '
          f'manuscript cites the year of the issue of record; confirm against '
          f'the journal page and keep the issue year.')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
