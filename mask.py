import warnings

from typing import List
from .utils import calc_checksum


class MaskField:
    def __init__(self, length: int, is_checksum: bool = False):
        self.length = length
        self.is_checksum = is_checksum

    def __int__(self):
        return self.length

    def __add__(self, other):
        return other + int(self)

    def __radd__(self, other):
        return self.__add__(other)


class Mask:
    def __init__(self, path: List[MaskField]):
        self.path = path

    @property
    def length(self):
        return sum(self.path)

    def parse(self, row: str, verify_checksum: bool = True):
        if len(row) != self.length:
            warnings.warn(f'Mask with summary length {self.length} does not match row with length {len(row)}',
                          stacklevel=2)

        parts = []
        i = 0
        for field in self.path:
            part = row[i: i + field.length]

            if verify_checksum and field.is_checksum and part.isdigit():
                real_checksum = int(part) if part.strip() else 0
                checksum = calc_checksum(parts[-1])
                if checksum != real_checksum:
                    raise ValueError(f'Invalid checksum for part {parts[-1]!r}: {checksum} instead of {real_checksum}')

            if set(part) == {'<'}:
                part = ''

            parts.append(part)
            i += field.length

        return [part.strip('<') for part in parts]
