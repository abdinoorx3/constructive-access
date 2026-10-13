"""Validate the first-public editorial delta without rerunning scientific banks.

Run from any directory: python publication/validate_public.py --build-pdf
The unchanged verify.py and research/clean_reproduction.py still run the full
scientific reproduction. This narrower command makes no claim to repeat them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tex_tail(text, command):
    markers = list(re.finditer(r'(?m)^' + re.escape(command) + r'[ \t]*$', text))
    if len(markers) != 1:
        raise ValueError('Expected one active TeX marker: ' + command)
    return text[markers[0].start():]


def bindings(root):
    path = root/'PUBLICATION_MAP.json'
    mapping = json.loads(path.read_text())
    delta_path = root/'PUBLIC_EDITORIAL_DELTA.json'
    editorial = {row['path']: row for row in json.loads(delta_path.read_text())['changes']} if delta_path.exists() else {}
    def bound(relative, expected):
        actual = sha(root/relative)
        edit = editorial.get(relative)
        return actual == expected or bool(edit and edit['original_sha256'] == expected and edit['public_sha256'] == actual)
    hashes = {'PUBLICATION_MAP.json': sha(path)}
    for row in mapping['baseline_files']:
        p = root/row['public_path']
        actual = sha(p)
        edit = editorial.get(row['public_path'])
        declared_edit = bool(edit and edit['original_sha256'] == row['sha256'] and edit['public_sha256'] == actual)
        if row['baseline_sha256'] != row['sha256'] or (actual != row['sha256'] and not declared_edit):
            raise ValueError(f'Baseline byte binding failed: {row["baseline_path"]}')
        hashes[row['public_path']] = actual
    for relative, digest in mapping['editorial_inputs_sha256'].items():
        if sha(root/relative) != digest:
            raise ValueError(f'Editorial source changed after staging: {relative}')
        hashes[relative] = digest
    evidence = mapping['scientific_evidence']
    for key in ('verification', 'portability'):
        if not bound(evidence[key], evidence[key+'_sha256']):
            raise ValueError(f'Scientific evidence report drift: {key}')
        hashes[evidence[key]] = evidence[key+'_sha256']
    verification = json.loads((root/evidence['verification']).read_text())
    portability = json.loads((root/evidence['portability']).read_text())
    if (verification['status'] != 'passed' or not verification['source_unchanged'] or
            portability['status'] != 'passed' or not all(portability['comparisons'].values())):
        raise ValueError('Archived scientific evidence is not successful')
    by_original = {row['baseline_path']: row for row in mapping['baseline_files']}
    for collection in ('source_sha256', 'result_sha256'):
        for relative, digest in verification[collection].items():
            if relative not in by_original or by_original[relative]['baseline_sha256'] != digest:
                raise ValueError(f'Missing baseline scientific report mapping: {relative}')
    expected = json.loads((root/'research/calibration_workflow.json').read_text())['expected_stage_names']
    if ([row['name'] for row in verification['commands']] != expected or
            any(row['exit_code'] for row in verification['commands'])):
        raise ValueError('Archived scientific stage list does not match the verified workflow')
    evidence_root = (root/evidence['verification']).parent
    for row in verification['commands']:
        p = evidence_root/row['log']
        if not bound(str(p.relative_to(root)), row['log_sha256']):
            raise ValueError(f'Archived scientific log drift: {row["name"]}')
        hashes[str(p.relative_to(root))] = row['log_sha256']
    original_ref = root/by_original['manuscript/reference.tex']['public_path']
    old, new = original_ref.read_text(), (root/'manuscript/reference.tex').read_text()
    if tex_tail(old, '\\mainmatter') != tex_tail(new, '\\mainmatter'):
        raise ValueError('Scientific chapter/bibliography sequence changed')
    if tex_tail(old, '\\tableofcontents') != tex_tail(new, '\\tableofcontents'):
        raise ValueError('Conventions or scientific body changed')
    for patch in mapping['editorial_chapter_patches']:
        original = (root/by_original[patch['path']]['public_path']).read_text()
        current = (root/patch['path']).read_text()
        if original.count(patch['old']) != 1 or original.replace(patch['old'], patch['new'], 1) != current:
            raise ValueError('Undeclared chapter change: ' + patch['path'])
    if mapping['toolkit_code_or_metadata_changed'] or mapping['scientific_content_changed']:
        raise ValueError('This validation path permits an editorial-only delta')
    for relative in ('pyproject.toml', 'verify.py', 'research/clean_reproduction.py'):
        if by_original[relative]['public_path'] != relative:
            raise ValueError(f'Required full reproduction source was relocated: {relative}')
    wheel = mapping['wheel']
    if sha(root/wheel['path']) != wheel['sha256']:
        raise ValueError('Archived scientific-baseline wheel byte identity failed')
    if not wheel['path'].startswith('validation/scientific_baseline/files/dist/'):
        raise ValueError('The baseline installed-wheel evidence must remain archived separately')
    metadata = json.loads((root/'PUBLIC_RELEASE.json').read_text())
    if metadata['publication_status'] not in ('prepared_not_published', 'published_to_repository'):
        raise ValueError('Unknown publication status')
    if metadata['publication_status'] == 'published_to_repository' and not metadata['publication_actions_performed']:
        raise ValueError('Published metadata requires an explicit publication record')
    if (root/'README.md').read_bytes() != (root/'README.txt').read_bytes():
        raise ValueError('GitHub and reproduction README payloads differ')
    return mapping, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--build-pdf', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    mapping, before = bindings(root)
    if not args.build_pdf:
        print(json.dumps({'status': 'binding_check_passed', 'files_checked': len(before),
                          'pdf_rebuilt': False, 'scientific_simulations_rerun': False}))
        return
    logs = root/'logs/publication'
    logs.mkdir(parents=True, exist_ok=True)
    commands = []
    env = dict(os.environ)
    for number in range(1, 4):
        started = time.monotonic()
        argv = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'reference.tex']
        done = subprocess.run(argv, cwd=root/'manuscript', env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors='replace')
        log = logs/f'pdflatex_{number}.log'
        log.write_text(done.stdout)
        commands.append({'name': f'public_pdflatex_{number}', 'argv': argv,
                         'exit_code': done.returncode, 'elapsed_seconds': round(time.monotonic()-started, 6),
                         'log': str(log.relative_to(root)), 'log_sha256': sha(log)})
        if done.returncode:
            raise RuntimeError(f'Public PDF compilation failed: {log}')
    texlog = (root/'manuscript/reference.log').read_text(errors='replace')
    problems = [line for line in texlog.splitlines()
                if any(marker in line for marker in ('Overfull ', 'undefined references',
                   'undefined citations', 'LaTeX Error', 'multiply defined'))]
    if problems:
        raise ValueError('Public PDF typesetting problems: ' + '\n'.join(problems))
    import fitz
    with fitz.open(root/'manuscript/reference.pdf') as pdf:
        pages = len(pdf)
        front = '\n'.join(page.get_text() for page in list(pdf)[:12])
    if 'First public release 1.0' not in front or 'Abdinoor' not in front or '2D Labs' not in front:
        raise ValueError('Public cover identity is absent from the compiled PDF')
    _, after = bindings(root)
    if before != after:
        raise ValueError('Publication inputs changed during PDF compilation')
    report = {
        'status': 'passed', 'publication_status': 'prepared_not_published',
        'validation_kind': 'Editorial delta over a byte-bound completed scientific baseline',
        'scientific_baseline': mapping['scientific_evidence'],
        'scientific_simulations_rerun': False,
        'scientific_sources_and_results': 'Scientific code and result bytes are retained; declared publication presentation edits are bound in PUBLIC_EDITORIAL_DELTA.json when present.',
        'inputs_unchanged': before == after, 'input_sha256': after,
        'commands': commands, 'typesetting_problems': problems, 'pdf_pages': pages,
        'pdf_sha256': sha(root/'manuscript/reference.pdf'),
        'installed_wheel': mapping['wheel'],
        'full_reproduction_entrypoints': ['python verify.py', 'python research/clean_reproduction.py'],
        'limitations': [
            'This is local validation, not actual publication or external peer review.',
            'License readiness and third-party rights require their separate declared disposition.',
            'No priority claim, empirical confirmation, or new scientific computation is inferred from an editorial rebuild.']}
    (root/'PUBLIC_VALIDATION.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': 'passed', 'pdf_pages': pages, 'files_bound': len(after),
                      'scientific_simulations_rerun': False, 'publication_status': 'prepared_not_published'}))


if __name__ == '__main__':
    main()
