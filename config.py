from .mask import Mask, MaskField


first_line_mask = Mask([
    MaskField(1),
    MaskField(1),
    MaskField(3),
    MaskField(39),
])

second_line_mask = Mask([
    MaskField(9),
    MaskField(1, is_checksum=True),
    MaskField(3),
    MaskField(6),
    MaskField(1, is_checksum=True),
    MaskField(1),
    MaskField(6),
    MaskField(1, is_checksum=True),
    MaskField(14),
    MaskField(1, is_checksum=True),
    MaskField(1)
])
