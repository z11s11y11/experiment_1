#!/usr/bin/env python3
"""Map every 128-byte chunk of GEMM's FP16 A matrix to one of six HBM groups.

This reproduces FlashGPU-Sim's H100-style, non-power-of-two IPOLY path with
96 memory channels, two subpartitions per channel, and 16 channels per assumed
HBM stack. The simulator itself models channels, not physical HBM stacks.
The 96-channel count is fixed for this six-HBM example, even if the checked
config currently has a different -gpgpu_n_mem value. Other mapping options
are read from and checked against the H100 config.

Source: src/gpgpu-sim/addrdec.cc:128-224 and hashing.cc:161-188.
Run: python3 python/map_a_to_hbm.py
"""

import argparse
import csv
from pathlib import Path


MATRIX_ROWS = 2560
MATRIX_COLUMNS = 2560
ELEMENT_BYTES = 2  # FP16
CHUNK_BYTES = 128
MEMORY_CHANNELS = 80
# MEMORY_CHANNELS = 96  # Six assumed HBM stacks * 16 channels each
SUBPARTITIONS_PER_CHANNEL = 2
CHANNELS_PER_HBM = 16
VIRTUAL_SUBPARTITIONS = 1024
DRAMID_START_BIT = 9
DEFAULT_BASE_ADDRESS = 0xC00000000  # GLOBAL_HEAP_START
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/SM90_H100/gpgpusim.config"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "a_hbm_map_5.csv"

# One list per output bit of IPOLY(1024), copied from hashing.cc:167-186.
IPOLY_BIT_INPUTS = (
    (30, 28, 27, 26, 24, 22, 20, 18, 16, 12, 9, 3, 0),
    (31, 29, 28, 27, 25, 23, 21, 19, 17, 13, 10, 4, 1),
    (32, 30, 29, 28, 26, 24, 22, 20, 18, 14, 11, 5, 2),
    (33, 31, 30, 29, 27, 25, 23, 21, 19, 15, 12, 6, 3, 0),
    (34, 32, 31, 30, 28, 26, 24, 22, 20, 16, 13, 7, 4, 1),
    (35, 33, 32, 31, 29, 27, 25, 23, 21, 17, 14, 8, 5, 2),
    (36, 34, 33, 32, 30, 28, 26, 24, 22, 18, 15, 9, 6, 3),
    (37, 35, 34, 33, 31, 29, 27, 25, 23, 19, 16, 10, 7, 4),
    (38, 36, 35, 34, 32, 30, 28, 26, 24, 20, 17, 11, 8, 5),
    (39, 37, 36, 35, 33, 31, 29, 27, 25, 21, 18, 12, 9, 6),
)
IPOLY_MASKS = tuple(sum(1 << bit for bit in bits) for bits in IPOLY_BIT_INPUTS)


def read_config_options(path: Path) -> dict[str, str]:
    options = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-"):
            key, _, value = line.partition(" ")
            options[key] = value.strip()
    return options


def bank_bit_positions(mapping: str) -> tuple[int, ...]:
    prefix, separator, layout = mapping.partition(";")
    if prefix != f"dramid@{DRAMID_START_BIT}" or not separator:
        raise ValueError(f"Expected dramid@{DRAMID_START_BIT} mapping, got {mapping!r}")
    layout = layout.replace(".", "").replace("|", "").replace(" ", "")
    if len(layout) != 64:
        raise ValueError("The DRAM address mapping must describe 64 bits")
    positions = tuple(sorted(63 - i for i, char in enumerate(layout) if char.upper() == "B"))
    if not positions:
        raise ValueError("No bank bits in DRAM address mapping")
    return positions


def validate_config(path: Path) -> tuple[int, ...]:
    options = read_config_options(path)
    expected = {
        "-gpgpu_n_sub_partition_per_mchannel": "2",
        "-gpgpu_memory_partition_indexing": "2",
        "-gpgpu_mem_address_mask": "1",
        "-gpgpu_ipoly_non_power2_balanced": "2",
        "-gpgpu_ipoly_channel_stable_l2slice": "0",
    }
    for key, value in expected.items():
        if options.get(key) != value:
            raise ValueError(f"{path}: expected {key} {value}, got {options.get(key)!r}")
    return bank_bit_positions(options["-gpgpu_mem_addr_mapping"])


def ipoly_1024(higher_bits: int, seed: int) -> int:
    result = seed
    for output_bit, mask in enumerate(IPOLY_MASKS):
        result ^= ((higher_bits & mask).bit_count() & 1) << output_bit
    return result


def hbm_for_address(address: int, bank_positions: tuple[int, ...]) -> int:
    # Non-power-of-two branch of addrdec_tlx(): decode the channel first.
    address_above_chip = address >> DRAMID_START_BIT
    decoded_channel = address_above_chip % MEMORY_CHANNELS
    higher_bits = address_above_chip // MEMORY_CHANNELS
    rest_of_address = (higher_bits << DRAMID_START_BIT) | (
        address & ((1 << DRAMID_START_BIT) - 1)
    )
    # addrdec_packbits() packs bank bits from low to high. Only its low bit
    # seeds the two-subpartition IPOLY mapping.
    bank_low_bit = (rest_of_address >> bank_positions[0]) & 1
    seed = decoded_channel * SUBPARTITIONS_PER_CHANNEL + bank_low_bit
    virtual_subpartition = ipoly_1024(higher_bits, seed)
    subpartition = (
        virtual_subpartition * MEMORY_CHANNELS * SUBPARTITIONS_PER_CHANNEL
        // VIRTUAL_SUBPARTITIONS
    )
    channel = subpartition // SUBPARTITIONS_PER_CHANNEL
    return channel // CHANNELS_PER_HBM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-address", type=lambda value: int(value, 0),
                        default=DEFAULT_BASE_ADDRESS,
                        help="A's device base address; default: 0xC00000000")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="H100 config providing the address mapping and hash options")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV path")
    args = parser.parse_args()
    if args.base_address < 0 or args.base_address % CHUNK_BYTES:
        parser.error("--base-address must be nonnegative and 128-byte aligned")

    bank_positions = validate_config(args.config)
    chunks_per_row = MATRIX_COLUMNS * ELEMENT_BYTES // CHUNK_BYTES
    if MATRIX_COLUMNS * ELEMENT_BYTES % CHUNK_BYTES:
        raise ValueError("A row is not an exact multiple of 128 bytes")

    counts = [0] * (MEMORY_CHANNELS // CHANNELS_PER_HBM)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        for row in range(MATRIX_ROWS):
            row_base = args.base_address + row * MATRIX_COLUMNS * ELEMENT_BYTES
            values = [hbm_for_address(row_base + chunk * CHUNK_BYTES, bank_positions)
                      for chunk in range(chunks_per_row)]
            for hbm in values:
                counts[hbm] += 1
            writer.writerow(values)

    print(f"Wrote {MATRIX_ROWS} rows x {chunks_per_row} columns to {args.output}")
    print("128-byte chunks per HBM:", ", ".join(f"{i}={count}" for i, count in enumerate(counts)))


if __name__ == "__main__":
    main()
