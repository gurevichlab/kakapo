# kakapo

A spectrum matching tool for mass spectrometry data analysis.

## Overview

**Status**: Development/Alpha

This tool matches experimental mass spectrometry spectra (MGF format) against theoretical fragmentations derived from molecular structures (rBAN YAML/JSON format).

## Installation

### Development Setup

```bash
# Clone the repository and navigate to the project directory
cd /path/to/nerpa-ms-varquest

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Quick Start

```bash
kakapo -s path/to/spectra.mgf -str path/to/structures.yaml
```

### CLI Options

#### Required Inputs

- `-s, --spectra <PATH...>`  
  Path(s) to experimental MGF spectra files or directories containing MGF files.

- `-str, --structures, --structs <PATH...>`  
  Path(s) to rBAN YAML/JSON structure files or directories containing structure files.

#### Matching & Filtering Options

- `-t, --tolerance <TOL>`  
  Mass tolerance window (e.g., `20ppm`, `0.02da`, or `0.02` for Da). *(Default: `0.02da`)*

- `--min-tolerance <FLOAT>`  
  Minimum absolute mass tolerance in Daltons (only applied when PPM tolerance is specified). *(Default: `0.001`)*

- `--min-peaks <INT>`  
  Minimum number of peaks required to include an experimental spectrum. *(Default: `5`)*

- `-m, --mode <MODE>`  
  Modification matching mode. *(Default: `free-mod`)*  
  Choices:
  - `no-mod`: Unmodified structures only
  - `free-mod`: Single modification allowed on any monomer
  - `x-mod`: Single modification allowed only on X monomers
  - `two-mod`: Known modification on base monomer + residual on X monomer

- `--scoring <SCORING>`  
  Scoring mode used for matching spectra to structures. *(Default: `all-peaks`)*  
  Choices:
  - `all-peaks`: Total matched peaks across theoretical and experimental spectra
  - `exp-peaks`: Experimental matched peaks count
  - `intensity`: Sum of intensities of matched peaks

- `--no-filters`  
  Disable default quality and preprocessing filters (`matchms`) on experimental spectra.

- `--no-cache`  
  Disable disk caching of parsed/cleaned experimental spectra and structures.

- `--cache-dir <DIR>`  
  Directory where parsed/cleaned spectra and structures are cached. *(Default: `.kakapo_cache`)*

#### Output Options

- `-o, --output <PATH>`  
  Path to output file or directory. If a directory or omitted, saves to default directory (`kakapo_output`). Specify `-` to print results to stdout.

- `--stdout`  
  Force printing match results to stdout.

- `-f, --format <FMT...>`  
  One or more output formats. *(Default: `table`)*  
  Choices: `table`, `txt`, `csv`, `json`.

- `--max-hits-per-spectrum <INT>`  
  Maximum number of top hits to output per spectrum by score. Use `0` to show all hits. *(Default: `10`)*

- `--top-n <INT>`  
  Limit output to only top N match results by score.

#### Logging Options

- `-l, --log-file <PATH>`  
  Path to write the log file. Defaults to `kakapo.log` inside the output directory.

- `--no-log-file`  
  Disable writing logs to a file.

- `-v, --verbose`  
  Increase logging verbosity (can be specified multiple times, e.g., `-v`, `-vv`).

### Example Commands

```bash
# Basic run with MGF file and YAML structure file
kakapo -s data/spectra.mgf -str structures/rban_records.yaml

# Advanced run specifying mass tolerance, modification mode, scoring, and output formats
kakapo -s data/spectra.mgf \
       -str structures/rban_records.yaml \
       -t 0.01da \
       --mode no-mod \
       --scoring intensity \
       -f csv json \
       -o output_dir \
       -v
```

## Project Structure

```
src/
└── kakapo/          # Main package
    ├── main.py      # Core workflow orchestration
    ├── cli.py       # Command-line interface
    ├── config.py    # Configuration defaults and options
    ├── constants.py # Chemical & mass-spectrometry constants loader
    ├── constants/   # Data files (atomic masses, modifications, weights)
    ├── models.py    # Data structure definitions
    ├── io/          # Input/Output parsers and writers
    ├── core/        # Core algorithmic modules (fragmenter, matcher)
    └── utils/       # Utility helpers (logging, etc.)

tests/               # Unit tests
```

## Development

### Running Tests

Run unit tests and check test coverage:

```bash
pytest
```

### Linting and Formatting

Use `ruff` for code linting and formatting:

```bash
ruff check src tests
ruff format src tests
```

## Contributing

Contributions are welcome! Please ensure all unit tests pass (`pytest`) and code is clean (`ruff check`) before submitting pull requests.

## License

MIT License (see LICENSE file for details)
