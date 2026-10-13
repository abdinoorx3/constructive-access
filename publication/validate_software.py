"""Build and validate the licensed public wheel without scientific-code changes.

Uses only local build tools and already-installed third-party dependencies.
The scientific source and pyproject are copied to a temporary build tree.
Run from any directory: python publication/validate_software.py
"""
from __future__ import annotations
import base64
import csv
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LICENSE_FILES = ('LICENSE', 'COMMERCIAL_USE.txt', 'THIRD_PARTY_NOTICES.txt',
                 'ATTRIBUTION_INVENTORY.json', 'font_licenses/AMS-OFL.txt',
                 'font_licenses/DEJAVU-LICENSE.txt', 'font_licenses/GUST-FONT-LICENSE.txt',
                 'font_licenses/LPPL-1.3c.txt')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def wheel_files(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt wheel archive')
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel member names')
        return {name: archive.read(name) for name in names if not name.endswith('/')}


def record(files, name):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    for path, data in sorted(files.items()):
        if path != name:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            writer.writerow((path, 'sha256='+encoded, len(data)))
    writer.writerow((name, '', ''))
    return stream.getvalue().encode()


def licensed_wheel(source, destination):
    files = wheel_files(source)
    metadata_paths = [name for name in files if name.endswith('.dist-info/METADATA')]
    if len(metadata_paths) != 1:
        raise ValueError('Expected exactly one wheel metadata directory')
    metadata_path = metadata_paths[0]
    prefix = metadata_path.rsplit('/', 1)[0]
    for name in LICENSE_FILES:
        files[prefix+'/licenses/'+name] = (ROOT/name).read_bytes()
    header, separator, body = files[metadata_path].partition(b'\n\n')
    if not separator:
        # A metadata file with no description body may end immediately after
        # its final header; that is valid core metadata emitted by setuptools.
        header, separator, body = files[metadata_path].rstrip(b'\n'), b'\n\n', b''
    lines = [line for line in header.split(b'\n') if not line.startswith(b'License-File:')]
    lines.extend(('License-File: '+name).encode() for name in LICENSE_FILES)
    files[metadata_path] = b'\n'.join(lines)+separator+body
    record_path = prefix+'/RECORD'
    files[record_path] = record(files, record_path)
    # A fixed ZIP timestamp and sorted members remove build-time ZIP variation.
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 13, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return prefix


def nonlicense_metadata(data):
    parsed = BytesParser(policy=compat32).parsebytes(data)
    headers = []
    for key, value in parsed.items():
        if key.lower() in ('license-file', 'license-expression'):
            continue
        if key.lower() == 'dynamic':
            value = ', '.join(part.strip() for part in value.split(',')
                              if part.strip().lower() not in ('license-file', 'license-expression'))
            if not value:
                continue
        headers.append((key, value))
    return headers, parsed.get_payload()


def audit_payload(baseline, public):
    old, new = wheel_files(baseline), wheel_files(public)
    package_old = {name: data for name, data in old.items() if '.dist-info/' not in name}
    package_new = {name: data for name, data in new.items() if '.dist-info/' not in name}
    if package_old != package_new:
        changed = sorted(name for name in package_old.keys() | package_new.keys()
                         if package_old.get(name) != package_new.get(name))
        raise ValueError('Scientific package payload changed: '+repr(changed))
    old_metadata = next(name for name in old if name.endswith('.dist-info/METADATA'))
    new_metadata = next(name for name in new if name.endswith('.dist-info/METADATA'))
    if nonlicense_metadata(old[old_metadata]) != nonlicense_metadata(new[new_metadata]):
        raise ValueError('Core package metadata changed beyond the license fields')
    prefix = new_metadata.rsplit('/', 1)[0]
    for name, data in old.items():
        if '.dist-info/' in name and not name.endswith(('/METADATA', '/RECORD')) and '/licenses/' not in name:
            if new.get(name) != data:
                raise ValueError('Nonlicense wheel metadata differs: '+name)
    licenses = {}
    for relative in LICENSE_FILES:
        name = prefix+'/licenses/'+relative
        expected = (ROOT/relative).read_bytes()
        if new.get(name) != expected:
            raise ValueError('License or notice payload differs: '+relative)
        licenses[relative] = digest(expected)
    parsed = BytesParser(policy=compat32).parsebytes(new[new_metadata])
    if sorted(parsed.get_all('License-File', [])) != sorted(LICENSE_FILES):
        raise ValueError('Core metadata fails to declare all bundled license files')
    record_path = prefix+'/RECORD'
    if new[record_path] != record(new, record_path):
        raise ValueError('Wheel RECORD does not bind the final public payload')
    return {'all_package_files_byte_identical': True,
            'package_files': len(package_old),
            'python_files': sum(name.endswith('.py') for name in package_old),
            'package_sha256': {name: digest(data) for name, data in sorted(package_old.items())},
            'nonlicense_core_metadata_unchanged': True,
            'other_nonlicense_distribution_metadata_unchanged': True,
            'license_payload_sha256': licenses, 'record_verified': True}


def main():
    mapping = json.loads((ROOT/'PUBLICATION_MAP.json').read_text())
    baseline = ROOT/mapping['wheel']['path']
    if sha(baseline) != mapping['wheel']['sha256']:
        raise ValueError('Archived baseline wheel binding failed')
    source_before = {str(p.relative_to(ROOT)): sha(p)
                     for p in (ROOT/'src').rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts)}
    source_before['pyproject.toml'] = sha(ROOT/'pyproject.toml')
    licensing_before = {name: sha(ROOT/name) for name in LICENSE_FILES}
    work = Path(tempfile.mkdtemp(prefix='constructive-access-public-wheel-'))
    logs = ROOT/'logs/public_software'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONOPTIMIZE'):
        env.pop(key, None)
    env.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    commands = []

    def run(name, argv, cwd=work):
        started = time.monotonic()
        process = subprocess.run(list(map(str, argv)), cwd=cwd, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors='replace')
        log = logs/(name+'.log')
        log.write_text(process.stdout)
        commands.append({'name': name, 'argv': list(map(str, argv)), 'exit_code': process.returncode,
                         'elapsed_seconds': round(time.monotonic()-started, 6),
                         'log': str(log.relative_to(ROOT)), 'log_sha256': sha(log)})
        if process.returncode:
            raise RuntimeError(name+' failed; inspect '+str(log))
        print(name+': PASS', flush=True)

    build = work/'build_source'
    build.mkdir()
    shutil.copy2(ROOT/'pyproject.toml', build/'pyproject.toml')
    shutil.copy2(ROOT/'LICENSE', build/'LICENSE')
    shutil.copytree(ROOT/'src', build/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    raw_dir = work/'raw_wheel'
    run('offline_build', [sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                          '--no-index', '--wheel-dir', raw_dir, build])
    raw_wheels = list(raw_dir.glob('*.whl'))
    if len(raw_wheels) != 1:
        raise ValueError('Expected one locally built raw wheel')
    public = work/raw_wheels[0].name
    licensed_wheel(raw_wheels[0], public)
    repeated = work/'repeat.whl'
    licensed_wheel(raw_wheels[0], repeated)
    if public.read_bytes() != repeated.read_bytes():
        raise ValueError('Packaging-only license normalization is not deterministic')
    payload = audit_payload(baseline, public)
    venv = work/'environment'
    run('fresh_venv', [sys.executable, '-m', 'venv', '--system-site-packages', venv])
    python = venv/'bin/python'
    run('offline_install', [python, '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--ignore-installed', public])
    run('installed_import', [python, '-I', '-c',
         "import constructive_access, pathlib, sys; p=pathlib.Path(constructive_access.__file__).resolve(); assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()), p; assert constructive_access.__version__=='0.10.0'; print(p)"])
    examples = work/'examples'
    examples.mkdir()
    for name in ('calibration_tutorial.py', 'calibration_counts.json'):
        shutil.copy2(ROOT/'examples'/name, examples/name)
    tutorial_output = work/'calibration_tutorial.json'
    run('installed_tutorial', [python, '-I', examples/'calibration_tutorial.py', '--output', tutorial_output])
    if tutorial_output.read_bytes() != (ROOT/'results/calibration_tutorial.json').read_bytes():
        raise ValueError('Public installed-wheel tutorial differs from the verified scientific report')
    run('installed_license_payload', [python, '-I', '-c',
         "import importlib.metadata,hashlib,json; d=importlib.metadata.distribution('constructive-access'); e=json.loads(" +
         repr(json.dumps(payload['license_payload_sha256'])) +
         "); f={str(x).split('.dist-info/licenses/',1)[1]:d.locate_file(x) for x in d.files if '.dist-info/licenses/' in str(x)}; assert all(k in f and hashlib.sha256(f[k].read_bytes()).hexdigest()==v for k,v in e.items()); print(len(e))"])
    for name, expected in {**source_before, **licensing_before}.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Scientific or licensing input changed during packaging: '+name)
    if sha(baseline) != mapping['wheel']['sha256']:
        raise ValueError('Archived scientific wheel changed during validation')
    run('public_bindings', [sys.executable, ROOT/'publication/validate_public.py'], ROOT)
    dist = ROOT/'dist'
    dist.mkdir(exist_ok=True)
    final = dist/public.name
    shutil.copy2(public, final)
    report = {
        'status': 'passed', 'publication_status': 'prepared_not_published',
        'validation_kind': 'Packaging-only license addition and fresh offline installation',
        'network_used': False, 'full_scientific_simulations_rerun': False,
        'environment': 'Temporary build source and fresh venv outside the public source tree; only preinstalled third-party dependencies are shared through system-site-packages. No PYTHONPATH; isolated Python import/tutorial execution.',
        'python': platform.python_version(),
        'build_tools': {name: importlib.metadata.version(name) for name in ('pip', 'setuptools', 'wheel')},
        'workspace': str(work), 'commands': commands,
        'baseline_wheel': {'path': str(baseline.relative_to(ROOT)), 'sha256': sha(baseline)},
        'public_wheel': {'path': str(final.relative_to(ROOT)), 'sha256': sha(final)},
        'raw_build_sha256': sha(raw_wheels[0]), 'normalized_build_repeated_byte_identically': True,
        'payload_comparison': payload,
        'source_and_pyproject_unchanged': True, 'scientific_source_sha256': source_before,
        'licensing_inputs_unchanged': True, 'license_input_sha256': licensing_before,
        'tutorial_byte_identical': True, 'tutorial_sha256': sha(tutorial_output),
        'script_sha256': sha(Path(__file__)),
        'reproduce': 'python publication/validate_software.py',
        'scope': 'Confirms packaging, notice inclusion and unchanged executable payload; it neither revalidates all science nor provides legal advice, external peer review or actual publication.'}
    (ROOT/'PUBLIC_SOFTWARE_VALIDATION.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': 'passed', 'package_files_identical': payload['package_files'],
                      'python_files_identical': payload['python_files'], 'license_files': len(LICENSE_FILES),
                      'public_wheel_sha256': sha(final), 'tutorial_byte_identical': True}))


if __name__ == '__main__':
    main()
