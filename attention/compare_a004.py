"""Bounded-memory comparison of canonically framed A004 JSON archives."""
import argparse
import gzip
import io
import json
import math
import re
from itertools import zip_longest
from pathlib import Path
from attention.a004 import decision_view
from attention.io import source_path, write_json
from scripts.compare_experiment_runs import numeric_summary

# The experiment writer uses indent=2. Nested values are fully parsed as JSON;
# only the root and package framing rely on that documented writer contract.
HEADER = re.compile(r'^  ("(?:[^"\\]|\\.)*"): (.*)\n$')
MAX_VALUE_CHARS = 256 * 1024 * 1024
MAX_METADATA_CHARS = 64 * 1024 * 1024


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _loads(text):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite JSON number: ' + value)
        return number
    return json.loads(text, object_pairs_hook=_object, parse_constant=invalid, parse_float=finite_float)


class Archive:
    """Yield packages once, retaining only small root metadata.

    Accepted framing is the project's indent=2 JSON writer: a root object,
    two-space root members and four-space object entries in packages. A
    different framing is rejected rather than guessed. Complete nested JSON
    values and every final delimiter are validated, including gzip integrity.
    """
    def __init__(self, path):
        self.path = source_path(path)
        self.metadata = {}
        self.package_count = 0
        self.finished = False
        self.metadata_chars = 0

    def _line(self, stream):
        line = stream.readline(MAX_VALUE_CHARS + 1)
        if not line:
            raise ValueError('Truncated JSON archive')
        if len(line) > MAX_VALUE_CHARS:
            raise ValueError('JSON line exceeds bounded reader limit')
        return line

    def _value(self, stream, first, indent):
        """Read a framed scalar/container, returning its parsed value and comma."""
        first = first.removesuffix('\n')
        if first in ('{', '['):
            closing = '}' if first == '{' else ']'
            tail = ' ' * indent + closing
            buffer = io.StringIO()
            buffer.write(first + '\n')
            size = len(first) + 1
            while True:
                line = self._line(stream)
                size += len(line)
                if size > MAX_VALUE_CHARS:
                    raise ValueError('JSON value exceeds bounded reader limit')
                if line in (tail + '\n', tail + ',\n'):
                    comma = line.endswith(',\n')
                    buffer.write(line[:-2] if comma else line[:-1])
                    raw = buffer.getvalue()
                    break
                buffer.write(line)
        else:
            comma = first.endswith(',')
            raw = first[:-1] if comma else first
            size = len(raw)
        return _loads(raw), comma, size

    def packages(self):
        opener = gzip.open if self.path.suffix == '.gz' else open
        try:
            with opener(self.path, 'rt', encoding='utf-8', newline='') as stream:
                if self._line(stream) != '{\n':
                    raise ValueError('Archive is not in canonical root framing')
                seen = set()
                need_member = False
                while True:
                    line = self._line(stream)
                    if line == '}\n':
                        if need_member:
                            raise ValueError('Trailing comma in root object')
                        if stream.read(1):
                            raise ValueError('Trailing content after JSON archive')
                        if 'packages' not in seen:
                            raise ValueError('Archive lacks packages')
                        self.finished = True
                        return
                    if seen and not need_member:
                        raise ValueError('Missing comma between root members')
                    match = HEADER.fullmatch(line)
                    if match is None:
                        raise ValueError('Archive is not in canonical member framing')
                    key, first = _loads(match.group(1)), match.group(2)
                    if key in seen:
                        raise ValueError('Duplicate root key: ' + key)
                    seen.add(key)
                    if key != 'packages':
                        value, need_member, size = self._value(stream, first, 2)
                        self.metadata_chars += size
                        if self.metadata_chars > MAX_METADATA_CHARS:
                            raise ValueError('Root metadata exceeds bounded reader limit')
                        self.metadata[key] = value
                        continue
                    if first in ('[]', '[],'):
                        need_member = first.endswith(',')
                        continue
                    if first != '[':
                        raise ValueError('Packages must use canonical array framing')
                    package_comma = False
                    while True:
                        line = self._line(stream)
                        if line in ('  ]\n', '  ],\n'):
                            if package_comma:
                                raise ValueError('Trailing comma in packages')
                            need_member = line.endswith(',\n')
                            break
                        if self.package_count and not package_comma:
                            raise ValueError('Missing comma between packages')
                        if line != '    {\n':
                            raise ValueError('Package must be a canonically framed object')
                        package, package_comma, _ = self._value(stream, '{', 4)
                        self.package_count += 1
                        yield package
        except (EOFError, UnicodeError, OSError, json.JSONDecodeError) as error:
            raise ValueError('Malformed or truncated archive: ' + str(error)) from error


def _package_view(package):
    wrapper = dict(intake_channel=[], global_permutations=[], intake_sha256='',
                   self_model_sha256='', packages=[package])
    return decision_view(wrapper)['packages'][0]


def compare_paths(first, second):
    left, right = Archive(first), Archive(second)
    missing = object()
    mismatches, diagnostics = [], {}
    same = True
    for index, (a, b) in enumerate(zip_longest(left.packages(), right.packages(), fillvalue=missing)):
        if a is missing or b is missing:
            same = False
            continue
        equal = _package_view(a) == _package_view(b)
        same = same and equal
        if not equal:
            mismatches.append(index)
        for path, row in numeric_summary([a], [b]).items():
            target = diagnostics.setdefault(path, dict(count=0, changed=0, max_abs=0., max_rel=0., nonfinite=0))
            for key in ('count', 'changed', 'nonfinite'):
                target[key] += row[key]
            for key in ('max_abs', 'max_rel'):
                target[key] = max(target[key], row[key])
    if not left.finished or not right.finished:
        raise ValueError('Archive reader did not reach a validated end')
    left.metadata['packages'] = []
    right.metadata['packages'] = []
    try:
        same = (decision_view(left.metadata) == decision_view(right.metadata)) and same
    except KeyError as error:
        raise ValueError('Missing required comparison metadata: ' + str(error)) from error
    return dict(decisions_pass=same, differing_packages=mismatches,
                numeric_diagnostics=diagnostics, numeric_diagnostics_informational=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('repeat', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite comparison output')
    result = compare_paths(args.original, args.repeat)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['decisions_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
