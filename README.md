WIP!!
Created 90% with copilot, may be broken, may be very bloated, but it mostly functons. 

# Stable Diffusion Model Sorter & Deduplicator

A comprehensive Python pipeline for organizing, deduplicating, and managing stable diffusion model files with intelligent metadata handling and advanced version detection.

## 🚀 Key Features

### Core Functionality
- **🔍 Advanced File Scanning**: SHA256 hash calculation with resume capability and intelligent file classification
- **📊 Comprehensive Metadata Extraction**: Extracts from SafeTensors headers, .civitai.info, .metadata.json files
- **🔗 Smart Duplicate Detection**: SHA256-based detection with sophisticated scoring system
- **📁 Intelligent Organization**: Structured sorting into `loras/base_model/model_name/` directories
- **🎯 Associated File Management**: Handles previews, metadata, and related files with enhanced version detection
- **🏷️ Civitai.info Generation**: Creates properly formatted civitai.info files from available metadata
- **💾 Database Integration**: Integrates with civitai.sqlite database for authoritative model information
- **🔄 Batch Processing**: Commit operations every 100 associations to prevent data loss
- **🧪 Dry Run Mode**: Complete preview functionality without file modifications

### Enhanced Features (2025 Updates)
- **🎨 Comprehensive Model Support**: All 10 model extensions (.safetensors, .ckpt, .pt, .pth, .bin, .vae, .onnx, .gguf, .safe, .oldsafetensors)
- **🔄 Advanced Version Detection**: Complex patterns (V11_0.85_V8_0.15), semantic versions (v1.5, v2.1), adaptive matching
- **🔗 Rescan-Linked Command**: Re-associate files with enhanced version detection and batch commits
- **🛡️ Robust Error Handling**: Read-only database support, graceful degradation, compound extension handling
- **📈 Progress Reporting**: Real-time progress updates, batch commit notifications, comprehensive statistics
- **🏗️ Modular Architecture**: Clean separation of concerns across 6 core modules

## Directory Structure

After processing, models are organized as:
```
sorted-models/
├── loras/
│   ├── SD 1.5/
│   │   ├── model_name_1/
│   │   │   ├── model.safetensors
│   │   │   ├── model.civitai.info
│   │   │   ├── model.preview.png
│   │   │   └── model.webp
│   │   └── model_name_2/
│   ├── SD 3.5/
│   ├── SDXL/
│   ├── Pony/
│   └── duplicates/
│       ├── duplicate_model_1/
│       └── duplicate_model_2/
├── checkpoint/
├── embedding/
└── other/
```

## 📦 Installation & Requirements

### System Requirements
- **Python 3.7+** (tested with Python 3.11+)
- **SQLite3** support (included with Python)
- **Standard Library Only** - no external dependencies required
- **Storage**: Sufficient space for database files and organized output
- **Memory**: 4GB+ recommended for large collections (>10,000 models)

### Quick Setup
1. **Clone or download** this repository
2. **Verify Python version**: `python3 --version` (should be 3.7+)
3. **Check core files** are present:
   ```bash
   ls -la *.py config.ini
   # Should show: model_sorter_main.py, file_scanner.py, metadata_extractor.py,
   #              duplicate_detector.py, model_sorter.py, civitai_generator.py
   ```
4. **Configure paths** in `config.ini`
5. **Test installation**: `python3 model_sorter_main.py --help`

### Optional Enhancements
- **Civitai Database**: Place `civitai.sqlite` in `Database/` directory for enhanced model detection
- **Server Database**: Configure server database path for batch processing capabilities

## Configuration

Edit `config.ini` to configure paths and settings:

```ini
[Paths]
source_directory = /path/to/source/models/
destination_directory = /path/to/sorted/models/
database_path = Database/civitai.sqlite

[Sorting]
use_model_type_subfolders = true
model_type_folders = checkpoint,loras,embedding,textualinversion,hypernetwork,controlnet,other
sanitize_folder_names = true
max_folder_name_length = 100

[Processing]
# All 10 supported model extensions (enhanced 2025)
model_extensions = .safetensors,.ckpt,.pt,.pth,.bin,.vae,.onnx,.gguf,.safe,.oldsafetensors
related_extensions = .preview.png,.webp,.jpg,.jpeg,.png,.civitai.info,.metadata.json,.txt,.yaml,.yml,.json
skip_existing_files = true
extract_related_folders = true
batch_commit_size = 100
enhanced_version_detection = true
server_database_path = /path/to/Stable-model-manager/model_sorter.sqlite
```

## 🚀 Usage

### Complete Workflow

Run the entire process (scan, extract metadata, detect duplicates, sort, generate civitai.info):

```bash
# Dry run (see what would happen without making changes)
python3 model_sorter_main.py --dry-run

# Live run (actually move files)
python3 model_sorter_main.py

# With custom config file
python3 model_sorter_main.py --config /path/to/config.ini

# Verbose output with detailed logging
python3 model_sorter_main.py --verbose

# Process specific base models only
python3 model_sorter_main.py --base-model "SD 1.5" --verbose
```

### Enhanced Commands (2025 Updates)

```bash
# Re-associate files with enhanced version detection (recommended)
python3 model_sorter_main.py --rescan-linked --verbose

# Completely rebuild all file associations from scratch
python3 model_sorter_main.py --rescan-linked-rebuild --verbose

# Repair metadata/text file path mismatches intelligently 
python3 model_sorter_main.py --rescan-and-repair-metadata-text --verbose

# Force rescan all files ignoring database cache
python3 model_sorter_main.py --step scan --force-rescan --verbose

# Extract metadata from specific number of files
python3 model_sorter_main.py --extract-metadata-limit 1000

# Extract metadata from ALL files
python3 model_sorter_main.py --extract-metadata-all

# Extract metadata including already scanned files 
python3 model_sorter_main.py --extract-metadata-all-rescan

# Test version detection without modifications
python3 model_sorter_main.py --rescan-linked --dry-run --verbose
```

### Individual Steps (All 15 Available Steps)

Run specific steps of the workflow:

```bash
# Core workflow steps
python3 model_sorter_main.py --step scan              # Scan files and calculate hashes
python3 model_sorter_main.py --step metadata          # Extract metadata from models
python3 model_sorter_main.py --step duplicates        # Detect duplicates using SHA256
python3 model_sorter_main.py --step sort --dry-run    # Sort and move models 
python3 model_sorter_main.py --step civitai           # Generate missing civitai.info files

# Enhanced processing steps (2025)
python3 model_sorter_main.py --step rescan                           # Legacy rescan
python3 model_sorter_main.py --step extract-metadata                 # Enhanced metadata extraction
python3 model_sorter_main.py --step rescan-linked                    # Re-associate files (recommended)
python3 model_sorter_main.py --step rescan-linked-rebuild            # Rebuild all associations
python3 model_sorter_main.py --step rescan-and-repair-metadata-text  # Repair metadata paths

# Database maintenance steps
python3 model_sorter_main.py --step retry-failed        # Retry previously failed extractions
python3 model_sorter_main.py --step migrate-paths       # Migrate database paths (requires args)
python3 model_sorter_main.py --step recover-missing     # Recover missing files automatically
python3 model_sorter_main.py --step force-blake3        # Add BLAKE3 hashes to non-SafeTensors
python3 model_sorter_main.py --step update-tables       # Check database completeness
```

### Individual Tools

Run components separately for testing or specific operations:

```bash
# File scanning only (with all 10 model extensions)
python3 file_scanner.py

# Metadata extraction with enhanced parsing
python3 metadata_extractor.py

# Duplicate detection with scoring system
python3 duplicate_detector.py

# Model sorting with directory structure creation
python3 model_sorter.py --dry-run

# Civitai.info generation with intelligent defaults
python3 civitai_generator.py

# Database inspection and analysis
python3 database_inspector.py
```

### Advanced Operations & Flags

```bash
# All command-line options available
python3 model_sorter_main.py --help  # Show complete help

# Configuration and output control
python3 model_sorter_main.py --config custom_config.ini     # Use custom config
python3 model_sorter_main.py --verbose --step scan          # Verbose output
python3 model_sorter_main.py --quiet --step metadata        # Reduce verbosity
python3 model_sorter_main.py --dry-run --verbose            # Preview mode

# Advanced scanning options  
python3 model_sorter_main.py --force-rescan --step scan     # Ignore cache, rescan all
python3 model_sorter_main.py --no-skip-folders --step scan  # Don't skip completed folders
python3 model_sorter_main.py --folder-limit 100 --step scan # Limit to 100 folders per run

# Database operations with arguments
python3 model_sorter_main.py --migrate-paths "/old/path/" "/new/path/"  # Migrate paths
python3 model_sorter_main.py --migrate-tables               # Migrate table structure
python3 model_sorter_main.py --update-tables                # Check database completeness
python3 model_sorter_main.py --recover-missing              # Auto-recover missing files

# Metadata extraction control
python3 model_sorter_main.py --extract-metadata-limit 500   # Extract from 500 files
python3 model_sorter_main.py --extract-metadata-all         # Extract from all files
python3 model_sorter_main.py --extract-metadata-all-rescan  # Re-extract from all files
python3 model_sorter_main.py --retry-failed                 # Retry failed extractions

# Hash operations
python3 model_sorter_main.py --force-blake3                 # Add BLAKE3 hashes

# Development and testing
python3 model_sorter_main.py --version                      # Show version
python3 model_sorter_main.py --dry-run --verbose --step all # Test complete workflow

# Configuration validation
python3 -c "
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
print('✅ Configuration valid')
for section in config.sections():
    print(f'[{section}]')
    for key, value in config[section].items():
        print(f'  {key} = {value}')
"

# Database statistics and health check
python3 -c "
import sqlite3, os
try:
    conn = sqlite3.connect('model_sorter.sqlite')
    cursor = conn.cursor()
    
    # Get table counts
    tables = ['scanned_files', 'model_files', 'associated_files', 'duplicate_groups', 'processing_log']
    print('📊 Database Statistics:')
    for table in tables:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f'  {table}: {count:,} records')
        except:
            print(f'  {table}: Table not found')
    
    # Database size
    db_size = os.path.getsize('model_sorter.sqlite') / (1024**2)
    print(f'  Database size: {db_size:.1f} MB')
    
    conn.close()
    print('✅ Database healthy')
except Exception as e:
    print(f'❌ Database error: {e}')
"
```

## 🧩 System Architecture

### 1. **File Scanner** (`file_scanner.py`) - 315KB
**Core scanning engine with enhanced capabilities**
- Recursively scans source directory for all 10 model extensions
- SHA256 hash calculation with resume capability and progress tracking
- Intelligent file classification (model, image, text, metadata, other)
- Enhanced associated file detection with version pattern matching
- SQLite database storage with optimized queries and indexing
- Dynamic SQL generation based on available model extensions

### 2. **Metadata Extractor** (`metadata_extractor.py`) - 48KB  
**Advanced metadata parsing and normalization**
- SafeTensors header extraction with error handling
- Civitai.info and metadata.json parsing with validation
- Base model and model type identification using database patterns
- Trained words extraction and normalization
- Metadata priority system (civitai.info > metadata.json > embedded > filename)
- Unicode handling and path normalization

### 3. **Duplicate Detector** (`duplicate_detector.py`) - 22KB
**Sophisticated duplicate management**
- SHA256-based duplicate detection with conflict resolution
- Advanced scoring system for duplicate prioritization:
  - Civitai.info presence (+10 points)
  - Metadata.json availability (+5 points)
  - Trained words (+3 points)
  - Associated images (+2 each)
- Metadata merging from all duplicates into primary version
- Intelligent duplicate folder organization

### 4. **Model Sorter** (`model_sorter.py`) - 63KB
**Comprehensive file organization system**
- Structured directory creation (`loras/base_model/model_name/`)
- Associated file movement with enhanced version detection
- Filename conflict resolution and deduplication
- Batch processing with configurable commit intervals
- Subdirectory extraction and flattening
- Target path validation and sanitization

### 5. **Civitai Info Generator** (`civitai_generator.py`) - 28KB
**Intelligent civitai.info file creation**
- Generates properly formatted civitai.info files from available metadata
- Database lookup by SHA256 for authoritative information
- Intelligent model type and base model detection from filename patterns
- Synthetic ID generation for locally created entries
- Full civitai.info structure with stats, files, and metadata sections
- Compatibility with existing civitai.sqlite database (4.3GB available)

### 6. **Main Controller** (`model_sorter_main.py`) - 87KB
**Unified pipeline orchestration**
- Complete workflow coordination across all modules
- Enhanced command-line interface with verbose logging
- Batch commit system (every 100 associations) to prevent data loss
- Read-only database support with graceful degradation
- Comprehensive error handling and progress reporting
- Dry-run mode for safe testing and preview functionality

## Database Schema

The script creates `model_sorter.sqlite` with the following tables:

- **scanned_files**: All scanned files with hashes and metadata
- **model_files**: Model-specific information and paths
- **associated_files**: Files associated with models
- **duplicate_groups**: Groups of duplicate models
- **processing_log**: Operation history and errors

## Duplicate Handling

The system intelligently handles duplicates by:

1. **Detection**: Files with identical SHA256 hashes are flagged as duplicates
2. **Scoring**: Each duplicate is scored based on:
   - Presence of civitai.info file (+10 points)
   - Presence of metadata.json file (+5 points)  
   - Trained words available (+3 points)
   - Associated image files (+2 points each)
3. **Primary Selection**: Highest scoring file becomes the primary version
4. **Metadata Merging**: Metadata from all duplicates is merged into the primary
5. **Duplicate Moving**: Secondary copies moved to `loras/duplicates/model_name_N/`

## Metadata Sources Priority

Metadata is extracted and prioritized from:

1. **civitai.info files** (highest priority)
2. **metadata.json files** 
3. **Embedded SafeTensor metadata**
4. **Generated from filename** (fallback)

## File Association Logic

Associated files are found by:

- Same base filename with different extensions
- Files in subdirectories matching patterns:
  - `model_name_images/`, `model_name_files/`
  - `model_name-data/`, `images/`, etc.
- All associated files are moved to the same directory as the model

## Safety Features

- **Dry Run Mode**: Test all operations without making changes
- **Resume Capability**: Hash cache allows resuming interrupted scans
- **Conflict Resolution**: Handles filename conflicts automatically
- **Error Logging**: Comprehensive error tracking and reporting
- **Database Backup**: Operations are logged for audit trails

## Output and Reporting

The system generates:

- **Console Output**: Real-time progress and results
- **Processing Reports**: JSON reports with detailed statistics
- **Database Logs**: All operations tracked in database
- **Error Summaries**: Comprehensive error reporting

Example processing report:
```json
{
  "timestamp": "2024-10-10T15:13:50.123456",
  "duration_seconds": 0.08,
  "dry_run": false,
  "stats": {
    "files_scanned": 5,
    "models_found": 1,
    "duplicates_found": 0,
    "models_sorted": 1,
    "civitai_files_generated": 0,
    "errors": 0
  }
}
```

## 📁 Supported Formats

### Model Files (All 10 Extensions - Enhanced 2025)
- **`.safetensors`** (preferred, supports metadata extraction)
- **`.ckpt`** (legacy checkpoint format)
- **`.pt`, `.pth`** (PyTorch formats)
- **`.bin`** (Hugging Face binary format) 
- **`.vae`** (VAE model files)
- **`.onnx`** (ONNX runtime models)
- **`.gguf`** (quantized models)
- **`.safe`** (SafeTensors variant)
- **`.oldsafetensors`** (legacy SafeTensors)

### Associated Files & Enhanced Version Detection
- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`
- **Metadata**: `.civitai.info`, `.metadata.json`, `.txt`, `.yaml`, `.yml`
- **Preview Images**: `.preview.png`, `.preview.jpg`, `.webp`, etc.
- **Version Patterns**: `V11_0.85_V8_0.15`, `v1.5`, `v2.1`, `V23`, adaptive matching modes

### Base Model Detection (Intelligent Pattern Matching)
- **SD 1.5**: `sd15`, `sd1.5`, `stable_diffusion_15`
- **SDXL 1.0**: `sdxl`, `xl`, `stable_diffusion_xl`
- **SD 3**: `sd3`, `stable_diffusion_3`, `stablediffusion3`
- **SD 3.5**: `sd35`, `sd3.5`, `stable_diffusion_35`
- **Flux.1 D**: `flux`, `flux1`, `flux_dev`
- **Flux.1 S**: `flux_schnell`, `flux_s`
- **Pony**: `pony`, `ponydiffusion`
- **SDXL Lightning**: `lightning`
- **Playground v2**: `playground`
- **PixArt**: `pixart`

## 🎯 Enhanced Features (2025 Updates)

### Advanced Version Detection
The system now supports complex version patterns in filenames:
- **Complex Patterns**: `V11_0.85_V8_0.15`, `V23_SD15_FLUX_mix`  
- **Semantic Versions**: `v1.5`, `v2.1`, `model_v3.2`
- **Simple Versions**: `V23`, `V11`, `version2`
- **Adaptive Matching**: Intelligent pattern recognition with fallback modes

### Batch Commit System
**Prevents data loss during long operations:**
```bash
python3 model_sorter_main.py --rescan-linked --verbose
# Output shows regular commits:
# 💾 Committed 100 new associations (Total: 100)  
# 💾 Committed 100 new associations (Total: 200)
# 💾 Committed 100 new associations (Total: 300)
# 💾 Final commit: 47 new associations (Total: 347)
```

### Enhanced Model Support
**Comprehensive format coverage:**
- All 10 model extensions supported
- Dynamic SQL generation prevents hardcoded limitations
- Database analysis integration for real-world format discovery
- Compound extension handling (`.civitai.info`, `.metadata.json`)

### Robust Error Handling
- **Read-only Database**: Graceful degradation to analysis mode when write permissions unavailable
- **Network Issues**: Handles server database connectivity problems  
- **Corruption Recovery**: Automatic fallback when databases become unavailable
- **Progress Preservation**: Batch commits ensure minimal data loss on interruption

## 🏆 Best Practices

### Initial Setup
1. **📁 Backup First**: Always backup your model collection before running
2. **🧪 Test with Dry Run**: Use `--dry-run` to verify behavior before live runs  
3. **📏 Start Small**: Test with a subset of models first
4. **⚙️ Check Config**: Verify paths and settings in config.ini
5. **👀 Monitor Output**: Watch for errors and verify results with `--verbose`

### Production Use  
6. **💾 Database Maintenance**: Periodically clean up old scan data
7. **🔄 Regular Rescans**: Use `--rescan-linked` after adding new files
8. **📊 Progress Monitoring**: Batch commits show processing progress every 100 items
9. **🛡️ Error Recovery**: Check processing reports for any issues
10. **🔍 Verification**: Use database inspection tools to verify results

## 🔧 Troubleshooting

### Common Issues & Solutions

**❌ "Source directory not found"**
```bash
# Verify path configuration
python3 -c "
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
print('Source dir:', config.get('Paths', 'source_directory'))
import os
print('Exists:', os.path.exists(config.get('Paths', 'source_directory')))
"
```

**❌ "Permission denied"**  
```bash
# Check directory permissions
ls -la /path/to/source/directory
ls -la /path/to/destination/directory
# Ensure write permissions on destination and database directory
chmod 755 /path/to/destination/directory
```

**❌ "Hash calculation errors"**
```bash
# Check for file system issues
python3 model_sorter_main.py --step scan --dry-run --verbose
# Look for specific files causing issues in output
```

**❌ "Database errors"**
```bash
# Reset local database
rm -f model_sorter.sqlite
python3 model_sorter_main.py --step scan --dry-run

# Check database permissions
ls -la *.sqlite
chmod 644 *.sqlite
```

**⚠️ "Batch commit failures"**
```bash
# Check server database connectivity
python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('/path/to/Stable-model-manager/model_sorter.sqlite', timeout=10)
    print('✅ Server database accessible')
    conn.close()
except Exception as e:
    print('❌ Server database issue:', e)
"
```

### 📖 Complete Command Reference

#### Core Workflow Flags
- **`--dry-run, -n`**: Preview mode - show what would be done without making changes
- **`--verbose, -v`**: Enable detailed logging and progress reporting
- **`--quiet, -q`**: Reduce output verbosity for automated scripts
- **`--config, -c`**: Specify custom configuration file path
- **`--version`**: Display version information

#### Step Selection (Choose One)
- **`--step scan`**: Scan directories and calculate file hashes
- **`--step metadata`**: Extract metadata from model files
- **`--step duplicates`**: Detect and organize duplicate files
- **`--step sort`**: Move files to organized directory structure
- **`--step civitai`**: Generate missing civitai.info files
- **`--step rescan`**: Legacy rescan functionality
- **`--step extract-metadata`**: Enhanced metadata extraction
- **`--step rescan-linked`**: Re-associate files with enhanced detection (recommended)
- **`--step rescan-linked-rebuild`**: Rebuild all associations from scratch
- **`--step rescan-and-repair-metadata-text`**: Repair metadata file path issues
- **`--step retry-failed`**: Retry previously failed operations
- **`--step migrate-paths`**: Update database paths (requires --migrate-paths args)
- **`--step recover-missing`**: Auto-recover missing files
- **`--step force-blake3`**: Add BLAKE3 hashes to non-SafeTensors files
- **`--step update-tables`**: Check and fix database completeness

#### Scanning Control Flags
- **`--force-rescan, -f`**: Ignore database cache, rescan all files
- **`--skip-folders`**: Skip directories already scanned (default: enabled)
- **`--no-skip-folders`**: Force scan all directories regardless of cache
- **`--folder-limit N`**: Process only N folders per run (incremental processing)

#### Metadata Processing Flags
- **`--extract-metadata-limit N`**: Extract metadata from N files only
- **`--extract-metadata-all`**: Extract metadata from all files
- **`--extract-metadata-all-rescan`**: Re-extract metadata including already processed files
- **`--retry-failed`**: Retry files that previously failed metadata extraction

#### File Association Flags (2025 Enhanced)
- **`--rescan-linked`**: Verify and correct file associations (batch commits every 100)
- **`--rescan-linked-rebuild`**: Completely rebuild all associations from scratch
- **`--rebuild-links`**: [DEPRECATED] Use --rescan-linked-rebuild instead
- **`--rescan-and-repair-metadata-text`**: Intelligently repair metadata path mismatches

#### Database Maintenance Flags
- **`--migrate-paths OLD NEW`**: Update database paths from OLD prefix to NEW prefix
- **`--migrate-tables`**: Migrate model files between database tables
- **`--update-tables`**: Check database completeness and fix missing entries
- **`--recover-missing`**: Automatically find and recover missing files
- **`--force-blake3`**: Add BLAKE3 hashes to files missing them

#### Example Command Combinations
```bash
# Complete fresh scan with verbose output
python3 model_sorter_main.py --force-rescan --verbose

# Incremental processing (100 folders at a time)
python3 model_sorter_main.py --folder-limit 100 --step scan --verbose

# Repair and rebuild everything
python3 model_sorter_main.py --rescan-linked-rebuild --extract-metadata-all-rescan --verbose

# Database maintenance and cleanup
python3 model_sorter_main.py --update-tables --recover-missing --verbose

# Safe preview of complete workflow
python3 model_sorter_main.py --dry-run --verbose

# Migration from old to new path structure
python3 model_sorter_main.py --migrate-paths "/old/mount/path/" "/new/mount/path/" --verbose
```

### Enhanced Diagnostics (2025)

**🔍 Version Detection Issues**
```bash
# Test version detection patterns
python3 model_sorter_main.py --rescan-linked --dry-run --verbose 2>&1 | grep -i "version\|pattern\|match"
```

**📊 Processing Statistics**
```bash
# Check current database state
python3 -c "
import sqlite3
conn = sqlite3.connect('model_sorter.sqlite')
cursor = conn.cursor()

tables = ['scanned_files', 'model_files', 'associated_files', 'duplicate_groups', 'processing_log']
for table in tables:
    try:
        cursor.execute(f'SELECT COUNT(*) FROM {table}')
        count = cursor.fetchone()[0]
        print(f'{table}: {count:,} records')
    except:
        print(f'{table}: Not found')
conn.close()
"
```

### Performance Tips

- **SSD Storage**: Use SSD for hash cache and database for better performance
- **Batch Size**: Adjust `hash_cache_save_interval` for memory usage
- **Parallel Processing**: Run on systems with multiple CPU cores
- **Network Drives**: Avoid scanning over slow network connections

## Advanced Usage

### Custom Base Model Mapping

Edit `metadata_extractor.py` to customize base model detection:

```python
def normalize_base_model_name(self, base_model: str) -> str:
    # Add custom mappings here
    if 'custom_model' in base_model.lower():
        return "Custom Model"
    # ... existing mappings
```

### Custom File Extensions

Add new model or associated file extensions in config.ini:

```ini
model_extensions = .safetensors,.ckpt,.pt,.pth,.bin,.onnx,.custom
related_extensions = .civitai.info,.metadata.json,.txt,.custom_meta
```

### Integration with Other Tools

The database can be queried directly for integration:

```python
import sqlite3
conn = sqlite3.connect('model_sorter.sqlite')
cursor = conn.cursor()

# Get all processed models
cursor.execute("""
    SELECT mf.model_name, mf.base_model, mf.target_path
    FROM model_files mf 
    WHERE mf.status = 'moved'
""")
```

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**What this means:**
- ✅ **Free to use** for personal and commercial projects
- ✅ **Free to modify** and create derivative works
- ✅ **Free to distribute** and share with others
- ✅ **Free to fork** and improve upon
- ⚠️ **No warranty** - use at your own risk, always backup your data

## 🤝 Contributing

**Contributions are very welcome!** This project is designed to be community-driven and improved by users.

### How to Contribute
1. **Fork the repository** and create your feature branch
2. **Test thoroughly** with both `--dry-run` and live modes
3. **Follow existing code style** and documentation patterns
4. **Add comprehensive error handling** for new features
5. **Update README** for any new functionality
6. **Ensure backward compatibility** with existing databases
7. **Submit a Pull Request** with detailed description of changes

### Areas needing improvement
- Performance optimization for very large collections (>100k files)
- Additional model format support
- Enhanced metadata extraction capabilities
- Better duplicate detection algorithms
- UI/web interface development
- Cross-platform compatibility improvements

## 📋 Recent Updates & Changelog

### 2025 Major Enhancements
- **✅ Comprehensive Model Support**: Added support for all 10 model extensions including .vae, .gguf, .safe, .oldsafetensors
- **✅ Enhanced Version Detection**: Complex pattern matching for version strings in filenames
- **✅ Batch Commit System**: Automatic commits every 100 associations to prevent data loss
- **✅ Advanced Error Handling**: Read-only database support, graceful degradation, compound extension handling
- **✅ Database Integration**: Full integration with 4.3GB civitai.sqlite database for authoritative metadata
- **✅ Rescan-Linked Command**: Re-associate files with enhanced detection capabilities
- **✅ Intelligent Base Model Detection**: Pattern matching for SD 1.5, SDXL, SD 3, Flux, Pony, and more
- **✅ Comprehensive Logging**: Detailed progress reporting and statistics with verbose mode

### Architecture Improvements
- **Modular Design**: Clean separation across 6 core modules (563KB total)
- **Dynamic SQL Generation**: Eliminates hardcoded extension lists
- **Performance Optimization**: Batch processing and commit strategies
- **Error Recovery**: Robust handling of database and network issues

## 🆘 Support & Contributing

### Getting Help
For issues, questions, or feature requests:

1. **📖 Check Documentation**: Review this README and troubleshooting section
2. **🧪 Test First**: Run with `--dry-run --verbose` to diagnose issues
3. **📊 Gather Information**: When reporting issues, include:
   - Full error messages with stack traces
   - Configuration file (with paths sanitized)
   - Steps to reproduce the problem
   - System information (Python version, OS, available disk space)
   - Database statistics from diagnostic commands

### Development
**Contributing Guidelines:**
- Test thoroughly with both dry-run and live modes
- Follow existing code style and documentation patterns  
- Add comprehensive error handling for new features
- Update this README for any new functionality
- Ensure backward compatibility with existing databases

### System Status Check
```bash
# Quick health check script
python3 -c "
import os, sys
print('🔍 System Status Check')
print('='*30)
print(f'Python: {sys.version.split()[0]}')
print(f'Working directory: {os.getcwd()}')

required_files = [
    'model_sorter_main.py', 'file_scanner.py', 'metadata_extractor.py',
    'duplicate_detector.py', 'model_sorter.py', 'civitai_generator.py', 'config.ini'
]

missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print(f'❌ Missing files: {missing}')
else:
    print('✅ All core files present')
    
if os.path.exists('Database/civitai.sqlite'):
    size = os.path.getsize('Database/civitai.sqlite') / (1024**3)
    print(f'✅ Civitai database: {size:.1f}GB')
else:
    print('ℹ️  Civitai database not found (optional)')
"
```
