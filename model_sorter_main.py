#!/usr/bin/env python3
"""
Model Sorter - Stable Diffusion Model Organization Pipeline

Unified pipeline that:
 - Recursively scans source directories for model files (.safetensors, .ckpt, .pt, .pth, .bin, .vae, .onnx, .gguf, .safe, .oldsafetensors)
 - Calculates SHA256 hashes for duplicate detection across different file names
 - Extracts metadata from SafeTensors headers, .metadata.json, .civitai.info files
 - Identifies and resolves duplicates using scoring system (civitai.info > metadata.json > file path depth)
 - Organizes models into structured directories: loras/base_model/model_name/
 - Generates missing civitai.info files from metadata or civitai database lookups
 - Integrates with civitai.sqlite database for authoritative model type/base model detection
 - Supports dry-run mode with comprehensive reporting and console output capture
 - Handles associated files (previews, metadata, etc.) alongside model files
 - Never skips files - duplicates are moved to numbered subdirectories

Notes:
 - This script executes file operations by default.
 - Use --dry-run to scan and prepare operations without moving files.
 - Reports are saved to processing_report/ directory for review and verification.
 - Requires civitai.sqlite database in Database/ for enhanced model detection.

Usage examples:
  python model_sorter_main.py                              # Run full organization workflow
  python model_sorter_main.py --dry-run                    # Scan and prepare without moving files
  python model_sorter_main.py --step scan                  # Run only file scanning
  python model_sorter_main.py --step sort --dry-run        # Preview sorting step
  python model_sorter_main.py --config custom_config.ini   # Use custom configuration
"""

import argparse
import configparser
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from io import StringIO
from typing import Optional

# Import our modules
from file_scanner import FileScanner, DatabaseManager
from metadata_extractor import MetadataExtractor
from duplicate_detector import DuplicateDetector
from model_sorter import ModelSorter
from civitai_generator import CivitaiInfoGenerator


def auto_detect_project_path(config_path: str):
    """
    Automatically detect the project directory and update config.ini with project-path.
    This ensures the script works with relative paths regardless of where it's run from.
    """
    try:
        # Get the directory where the main script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load config
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Ensure [Paths] section exists
        if 'Paths' not in config:
            config.add_section('Paths')
        
        # Get current project-path from config
        current_project_path = config.get('Paths', 'project_path', fallback='')
        
        # Only update if project-path is not already set to the correct directory
        if current_project_path != script_dir:
            config.set('Paths', 'project_path', script_dir)
            
            # Write back to config file
            with open(config_path, 'w') as config_file:
                config.write(config_file)
            
            print(f"Auto-detected project directory: {script_dir}")
            print(f"Updated project-path in {config_path}")
        
    except Exception as e:
        print(f"Warning: Could not auto-detect project path: {e}")
        print("Please manually set 'project_path' in config.ini if needed")


class ConsoleCapture:
    """Capture console output for inclusion in reports"""
    
    def __init__(self):
        self.captured_output = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def start_capture(self):
        """Start capturing console output"""
        self.capture_buffer = StringIO()
        sys.stdout = TeeOutput(self.original_stdout, self.capture_buffer)
        sys.stderr = TeeOutput(self.original_stderr, self.capture_buffer)
        
    def stop_capture(self):
        """Stop capturing and return captured output"""
        if hasattr(self, 'capture_buffer'):
            sys.stdout = self.original_stdout  
            sys.stderr = self.original_stderr
            captured = self.capture_buffer.getvalue()
            self.capture_buffer.close()
            return captured.splitlines()
        return []


class TeeOutput:
    """Tee output to both original stream and capture buffer"""
    
    def __init__(self, original, capture):
        self.original = original
        self.capture = capture
        
    def write(self, text):
        self.original.write(text)
        self.capture.write(text)
        
    def flush(self):
        self.original.flush()
        self.capture.flush()


class ModelSorterOrchestrator:
    """Main orchestrator class that coordinates all components"""
    
    def __init__(self, config_path: str = "config.ini", dry_run: bool = False, 
                 verbose: bool = True, force_rescan: bool = False, extract_metadata_limit: Optional[int] = None,
                 extract_metadata_all: bool = False, extract_metadata_all_rescan: bool = False, retry_failed: bool = False,
                 skip_folders: bool = True, folder_limit: Optional[int] = None):
        self.config_path = config_path
        self.dry_run = dry_run
        self.verbose = verbose
        self.force_rescan = force_rescan
        self.extract_metadata_limit = extract_metadata_limit or 100  # Default to 100 if not specified
        self.extract_metadata_all = extract_metadata_all
        self.extract_metadata_all_rescan = extract_metadata_all_rescan
        self.retry_failed = retry_failed
        self.skip_folders = skip_folders
        self.folder_limit = folder_limit
        
        # Initialize console capture
        self.console_capture = ConsoleCapture()
        
        # Initialize components
        self.scanner = FileScanner(config_path, force_rescan=force_rescan, skip_folders=skip_folders, verbose=verbose)
        self.metadata_extractor = MetadataExtractor()
        self.duplicate_detector = DuplicateDetector()
        self.model_sorter = ModelSorter(config_path, dry_run)
        self.civitai_generator = CivitaiInfoGenerator()
        
        # Stats
        self.start_time = time.time()
        self.stats = {
            'files_scanned': 0,
            'models_found': 0,
            'duplicates_found': 0,
            'models_sorted': 0,
            'models_migrated': 0,
            'civitai_files_generated': 0,
            'errors': 0
        }
    
    def print_step(self, step_name: str, description: str = ""):
        """Print a step header"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STEP: {step_name}")
            if description:
                print(f"Description: {description}")
            print(f"{'='*60}")
    
    def print_progress(self, message: str):
        """Print progress message"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def load_config(self):
        """Load configuration from config file"""
        import configparser
        config = configparser.ConfigParser()
        config.read(self.config_path)
        return config
    
    def get_server_database_path(self) -> str:
        """Get server database path from config"""
        config = self.load_config()
        try:
            return config.get('Paths', 'server_database_path')
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Fallback to default if not configured - use generic path
            return "model_organizer/model_sorter.sqlite"
    
    def determine_model_type_from_extension(self, file_path: str) -> str:
        """Determine model type from file extension and path context"""
        file_path_lower = file_path.lower()
        
        if file_path_lower.endswith(('.safetensors', '.safe', '.oldsafetensors')):
            if 'checkpoint' in file_path_lower or 'model' in file_path_lower:
                return 'Checkpoint'
            else:
                return 'LORA'  # Default for .safetensors variants
        elif file_path_lower.endswith('.ckpt'):
            return 'Checkpoint'
        elif file_path_lower.endswith(('.pt', '.pth')):
            # .pt/.pth can be LoRA, TextualInversion, Hypernetwork, VAE, or Upscaler
            if 'vae' in file_path_lower:
                return 'VAE'
            elif 'upscal' in file_path_lower:
                return 'Upscaler'
            elif 'hypernetwork' in file_path_lower:
                return 'Hypernetwork'
            elif 'embedding' in file_path_lower or 'textualinversion' in file_path_lower:
                return 'TextualInversion'
            else:
                return 'LORA'  # Default for .pt/.pth
        elif file_path_lower.endswith('.bin'):
            return 'TextualInversion'  # Binary embeddings
        elif file_path_lower.endswith('.vae'):
            return 'VAE'  # Standalone VAE files
        elif file_path_lower.endswith('.onnx'):
            return 'Other'  # ONNX runtime models
        elif file_path_lower.endswith('.gguf'):
            return 'Other'  # Quantized models
        else:
            return 'LORA'  # Default fallback

    def step_1_scan_files(self) -> bool:
        """Step 1: Scan source directory for files and generate hashes"""
        try:
            # Detect if this is a fast prescan (dry-run mode for preliminary mapping)
            if self.dry_run:
                self.print_step("1. ENHANCED PRESCAN (DRY-RUN)", 
                              "Check all files, add missing ones to database with full metadata extraction")
                fast_prescan = True
            else:
                self.print_step("1. FILE SCANNING", 
                              "Scanning source directory and calculating SHA256 hashes")
                fast_prescan = False
            
            source_dir = self.scanner.config['source_directory']
            self.print_progress(f"Scanning directory: {source_dir}")
            
            if not os.path.exists(source_dir):
                print(f"ERROR: Source directory not found: {source_dir}")
                return False
            
            # Perform the scan (with fast prescan mode if dry-run and folder limit if specified)
            results = self.scanner.scan_directory(source_dir, fast_prescan=fast_prescan, force_blake3=self.force_rescan, folder_limit=self.folder_limit, verbose_dry_run=(self.dry_run or self.verbose))
            
            # Update stats
            self.stats['files_scanned'] = sum(len(files) for files in results.values())
            self.stats['models_found'] = len(results['models'])
            
            if fast_prescan:
                self.print_progress(f"Fast prescan complete: {self.stats['files_scanned']} unscanned files identified")
                self.print_progress(f"SafeTensors models needing full scan: {self.stats['models_found']}")
                self.print_progress("Use --step scan (without --dry-run) to perform full scanning")
            else:
                self.print_progress(f"Scan complete: {self.stats['files_scanned']} files processed")
                self.print_progress(f"Model files found: {self.stats['models_found']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in file scanning: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_2_extract_metadata(self) -> bool:
        """Step 2: Extract metadata from model files"""
        try:
            self.print_step("2. METADATA EXTRACTION", 
                          "Extracting metadata from model files and associated text files")
            
            self.print_progress("Processing scanned model files...")
            self.metadata_extractor.process_scanned_models()
            
            self.print_progress("Metadata extraction complete")
            return True
            
        except Exception as e:
            print(f"ERROR in metadata extraction: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_3_detect_duplicates(self) -> bool:
        """Step 3: Detect and handle duplicates"""
        try:
            self.print_step("3. DUPLICATE DETECTION", 
                          "Identifying duplicates and determining best versions to keep")
            
            self.print_progress("Analyzing files for duplicates...")
            duplicate_groups = self.duplicate_detector.create_duplicate_groups()
            
            summary = self.duplicate_detector.get_duplicate_summary()
            self.stats['duplicates_found'] = summary['duplicate_files']
            
            self.print_progress(f"Duplicate detection complete")
            self.print_progress(f"Duplicate groups: {summary['duplicate_groups']}")
            self.print_progress(f"Duplicate files: {summary['duplicate_files']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in duplicate detection: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_4_sort_models(self) -> bool:
        """Step 4: Sort and move model files using batch processing"""
        try:
            self.print_step("4. MODEL SORTING", 
                          "Moving models to organized directory structure using batch processing")
            
            if self.dry_run:
                self.print_progress("DRY RUN MODE - No files will actually be moved")
            
            # Use the new batch processing system from FileScanner
            self.print_progress("Sorting models using batch processing (batch size from config.ini)...")
            sort_results = self.scanner.sort_models_batch()
            
            # Update stats from batch results
            self.stats['models_sorted'] = sort_results.get('models_moved', 0)
            
            self.print_progress(f"Batch sorting complete")
            self.print_progress(f"Total models processed: {sort_results.get('total_models_processed', 0)}")
            self.print_progress(f"Models moved: {sort_results.get('models_moved', 0)}")
            self.print_progress(f"Associated files moved: {sort_results.get('associated_files_moved', 0)}")
            self.print_progress(f"Database batches committed: {sort_results.get('batches_committed', 0)}")
            if sort_results.get('errors', 0) > 0:
                self.print_progress(f"Errors encountered: {sort_results.get('errors', 0)}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in model sorting: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_5_generate_civitai_files(self) -> bool:
        """Step 5: Generate missing civitai.info files"""
        try:
            self.print_step("5. CIVITAI.INFO GENERATION", 
                          "Generating civitai.info files for models that lack them")
            
            if self.dry_run:
                self.print_progress("DRY RUN MODE - No civitai.info files will be generated")
                return True
            
            self.print_progress("Generating missing civitai.info files...")
            success_count, error_count = self.civitai_generator.process_models_needing_civitai_info()
            
            # Also generate for moved models
            self.civitai_generator.generate_missing_civitai_files_for_moved_models()
            
            self.stats['civitai_files_generated'] = success_count
            self.stats['errors'] += error_count
            
            self.print_progress(f"Civitai.info generation complete")
            self.print_progress(f"Files generated: {success_count}")
            if error_count > 0:
                self.print_progress(f"Errors: {error_count}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in civitai.info generation: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_rescan_metadata(self) -> bool:
        """Rescan step: Update missing image metadata in database"""
        try:
            self.print_step("RESCAN", 
                          "Re-scanning files to update missing image metadata in database")
            
            self.print_progress("Checking for files with missing image metadata...")
            update_stats = self.scanner.rescan_missing_metadata()
            
            self.stats['files_updated'] = update_stats['updated']
            
            self.print_progress(f"Image metadata rescan complete")
            self.print_progress(f"Files updated: {update_stats['updated']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in metadata rescan: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_extract_civitai_metadata(self) -> bool:
        """Extract enhanced Civitai metadata from existing files in database"""
        try:
            self.print_step("EXTRACT CIVITAI METADATA", 
                          "Scanning existing files to extract enhanced Civitai metadata")
            
            if self.extract_metadata_all_rescan:
                self.print_progress(f"Extracting Civitai metadata from ALL files (including already scanned)...")
                stats = self.scanner.scan_existing_files_for_metadata(None, force_rescan=True)
            elif self.extract_metadata_all:
                self.print_progress(f"Extracting Civitai metadata from ALL files needing processing...")
                stats = self.scanner.scan_existing_files_for_metadata(None)
            else:
                self.print_progress(f"Extracting Civitai metadata from up to {self.extract_metadata_limit} files...")
                stats = self.scanner.scan_existing_files_for_metadata(self.extract_metadata_limit)
            
            self.stats.update(stats)
            
            self.print_progress(f"Civitai metadata extraction complete")
            self.print_progress(f"Files processed: {stats['processed']}")
            self.print_progress(f"Files updated: {stats['updated']}")
            self.print_progress(f"Files skipped: {stats['skipped']}")
            self.print_progress(f"Errors: {stats['errors']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in Civitai metadata extraction: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_retry_failed_metadata(self) -> bool:
        """Step: Retry metadata extraction for previously failed files"""
        try:
            self.print_step("RETRY FAILED METADATA EXTRACTION", 
                           "Retrying files that previously failed due to missing database columns")
            
            self.print_progress(f"Retrying metadata extraction for previously failed files...")
            stats = self.scanner.retry_failed_metadata_extraction()
            
            self.print_progress(f"Retry metadata extraction complete")
            self.print_progress(f"Files processed: {stats['processed']}")
            self.print_progress(f"Files updated: {stats['updated']}")
            self.print_progress(f"Files skipped: {stats['skipped']}")
            self.print_progress(f"Errors: {stats['errors']}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in retry failed metadata extraction: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_migrate_paths(self, old_prefix: str, new_prefix: str) -> bool:
        """Step: Migrate database paths from old prefix to new prefix"""
        try:
            self.print_step("MIGRATE DATABASE PATHS", 
                           f"Migrating paths from '{old_prefix}' to '{new_prefix}'")
            
            self.print_progress(f"Migrating database paths...")
            results = self.scanner.migrate_database_paths(old_prefix, new_prefix, dry_run=self.dry_run)
            
            self.print_progress(f"Path migration complete")
            self.print_progress(f"Files found with old prefix: {results['files_found']}")
            self.print_progress(f"Files verified at new path: {results['files_verified']}")
            self.print_progress(f"Files missing at new path: {results['files_missing']}")
            
            if not self.dry_run:
                self.print_progress(f"Files updated in database: {results['files_updated']}")
            else:
                self.print_progress(f"[DRY RUN] Would update {results['files_verified']} file paths")
            
            if results['files_missing'] > 0:
                self.print_progress(f"WARNING: {results['files_missing']} files not found at new paths")
                self.print_progress("Consider checking if files were actually moved/renamed")
            
            return True
            
        except Exception as e:
            print(f"ERROR in path migration: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_recover_missing_files(self) -> bool:
        """Step: Recover missing files using automatic recovery strategies"""
        try:
            self.print_step("RECOVER MISSING FILES", 
                           "Automatically recovering missing files using alternative paths and destinations")
            
            self.print_progress(f"Scanning for missing files and attempting recovery...")
            results = self.scanner.bulk_recover_missing_files(dry_run=self.dry_run, limit=200)
            
            self.print_progress(f"File recovery complete")
            self.print_progress(f"Files checked: {results['total_checked']}")
            self.print_progress(f"Files recovered: {results['recovered']}")
            self.print_progress(f"Files still missing: {results['still_missing']}")
            
            if not self.dry_run:
                self.print_progress(f"Database updates: {results['db_updates']}")
            else:
                self.print_progress(f"[DRY RUN] Would update {results['recovered']} file paths")
            
            if results['recovered'] > 0:
                self.print_progress(f"Successfully recovered {results['recovered']} missing files!")
                
            if results['still_missing'] > 0:
                self.print_progress(f"WARNING: {results['still_missing']} files could not be recovered")
            
            return True
            
        except Exception as e:
            print(f"ERROR in file recovery: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_force_blake3_rescan(self) -> bool:
        """Step: Force rescan of non-SafeTensors files to add BLAKE3 hashes"""
        try:
            self.print_step("FORCE BLAKE3 RESCAN", 
                           "Rescanning non-SafeTensors files to add fast BLAKE3 hashes")
            
            source_dir = self.scanner.config['source_directory']
            self.print_progress(f"Scanning directory for non-SafeTensors files: {source_dir}")
            
            if not os.path.exists(source_dir):
                print(f"ERROR: Source directory not found: {source_dir}")
                return False
            
            # Perform BLAKE3 rescan (force_blake3=True for non-SafeTensors)
            self.scanner.force_rescan = True  # Force rescanning
            results = self.scanner.scan_directory(source_dir, fast_prescan=False, force_blake3=True)
            
            # Update stats
            self.stats['files_scanned'] = sum(len(files) for files in results.values())
            self.stats['models_found'] = len(results['models'])
            
            self.print_progress(f"BLAKE3 rescan complete: {self.stats['files_scanned']} files processed")
            self.print_progress(f"Files updated with BLAKE3 hashes: {self.stats['models_found']}")
            self.print_progress("Non-SafeTensors files now have fast BLAKE3 hashes for future operations")
            
            return True
            
        except Exception as e:
            print(f"ERROR in BLAKE3 force rescan: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_update_tables(self) -> bool:
        """Step: Check database completeness and rescan files with missing table entries"""
        try:
            self.print_step("UPDATE TABLES", 
                           "Checking database completeness and rescanning files with missing entries")
            
            source_dir = self.scanner.config['source_directory']
            self.print_progress(f"Analyzing database completeness for: {source_dir}")
            
            if not os.path.exists(source_dir):
                print(f"ERROR: Source directory not found: {source_dir}")
                return False
            
            # Get files that need rescanning due to incomplete database entries
            incomplete_files = self.scanner.db.get_files_needing_rescan()
            
            if not incomplete_files:
                self.print_progress("✓ All files in database have complete table entries")
                return True
            
            self.print_progress(f"Found {len(incomplete_files)} files with incomplete table entries")
            
            # Rescan incomplete files using existing metadata scanning functionality
            scan_results = self.scanner.scan_existing_files_for_metadata(
                limit=None,  # No limit, scan all incomplete files
                force_rescan=True
            )
            
            # Update stats
            self.stats['files_scanned'] = scan_results.get('processed', 0)
            
            self.print_progress(f"Database update complete:")
            self.print_progress(f"  - Files processed: {scan_results.get('processed', 0)}")
            self.print_progress(f"  - Files updated: {scan_results.get('updated', 0)}")
            self.print_progress(f"  - Files skipped: {scan_results.get('skipped', 0)}")
            self.print_progress(f"  - Errors encountered: {scan_results.get('errors', 0)}")
            
            if scan_results.get('errors', 0) > 0:
                self.stats['errors'] += scan_results['errors']
                self.print_progress("Some files had errors during rescanning")
            else:
                self.print_progress("All incomplete files have been successfully rescanned")
            
            return scan_results.get('errors', 0) == 0
            
        except Exception as e:
            print(f"ERROR in database update: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_rescan_linked_files(self) -> bool:
        """Step: Verify and correct file associations (incremental check, not complete rebuild)"""
        try:
            self.print_step("VERIFY LINKED FILES", 
                           "Verifying and correcting file associations without complete rebuild")
            
            # Use server database from config for this operation
            config = self.load_config()
            server_db_path = config.get('Paths', 'server_database_path', fallback='model_sorter.sqlite')
            
            # Create scanner instance with server database for correlation logic
            from file_scanner import DatabaseManager, FileScanner
            temp_scanner = FileScanner(self.config_path)
            temp_scanner.db = DatabaseManager(server_db_path)
            temp_scanner.config['verbose'] = self.verbose
            conn = temp_scanner.db.conn
            
            if self.verbose:
                print(f"🔗 Using configured database: {server_db_path}")
            
            cursor = conn.cursor()
            
            # Get models that have metadata but missing associations
            cursor.execute('''
                SELECT mf.id, mf.scanned_file_id, sf.file_path, sf.file_name
                FROM model_files mf
                JOIN scanned_files sf ON mf.scanned_file_id = sf.id
                LEFT JOIN (
                    SELECT model_file_id, COUNT(*) as assoc_count
                    FROM associated_files
                    GROUP BY model_file_id
                ) af ON mf.id = af.model_file_id
                WHERE af.assoc_count IS NULL OR af.assoc_count = 0
                ORDER BY sf.file_path
            ''')
            
            models_needing_associations = cursor.fetchall()
            self.print_progress(f"Found {len(models_needing_associations)} models without associations")
            
            if self.verbose:
                if len(models_needing_associations) > 0:
                    print(f"📋 Models needing associations:")
                    for i, (_, _, model_path, model_name) in enumerate(models_needing_associations[:5]):  # Show first 5
                        print(f"   {i+1}. {model_name}")
                    if len(models_needing_associations) > 5:
                        print(f"   ... and {len(models_needing_associations) - 5} more")
                    print("")
            
            # Check for orphaned associated files (files that could belong to models but aren't linked)
            cursor.execute('''
                SELECT sf.id, sf.file_path, sf.file_name
                FROM scanned_files sf
                LEFT JOIN associated_files af ON sf.id = af.scanned_file_id
                WHERE sf.file_type IN ('text', 'image', 'video', 'json', 'unknown')
                AND af.scanned_file_id IS NULL
                ORDER BY sf.file_path
            ''')
            
            orphaned_files = cursor.fetchall()
            self.print_progress(f"Found {len(orphaned_files)} potentially orphaned associated files")
            
            if self.verbose and len(orphaned_files) > 0:
                print(f"📋 Orphaned files found:")
                for i, (_, file_path, file_name) in enumerate(orphaned_files[:5]):  # Show first 5
                    print(f"   {i+1}. {file_name}")
                if len(orphaned_files) > 5:
                    print(f"   ... and {len(orphaned_files) - 5} more")
                print("")
            
            associations_created = 0
            missing_associations_found = 0
            
            # Initialize batch commit tracking
            associations_since_last_commit = 0
            COMMIT_BATCH_SIZE = 100  # Commit every 100 associations
            
            # Initialize unmatched files logging
            from datetime import datetime
            unmatched_log = []
            models_processed_count = 0
            log_filename = f"unmatched_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = os.path.join("processing_report", log_filename)
            
            # Ensure directory exists
            os.makedirs("processing_report", exist_ok=True)
            
            def commit_batch_associations():
                """Commit database changes and reset counter"""
                nonlocal associations_since_last_commit
                if associations_since_last_commit > 0:
                    try:
                        conn.commit()
                        if self.verbose:
                            print(f"💾 Committed {associations_since_last_commit} new associations to database (Total created: {associations_created})")
                        associations_since_last_commit = 0
                    except Exception as e:
                        if "readonly database" in str(e).lower():
                            if self.verbose:
                                print("   ℹ️ Database is read-only - skipping commit")
                            associations_since_last_commit = 0
                        else:
                            raise
            
            def write_unmatched_log_batch():
                """Write current unmatched log to file"""
                if unmatched_log:
                    import json
                    log_data = {
                        'timestamp': datetime.now().isoformat(),
                        'command': 'rescan-linked',
                        'batch_info': {
                            'models_processed_so_far': models_processed_count,
                            'total_models': len(models_needing_associations),
                            'unmatched_in_this_batch': len(unmatched_log)
                        },
                        'unmatched_files': unmatched_log
                    }
                    
                    # Append to existing file or create new
                    try:
                        if os.path.exists(log_path):
                            with open(log_path, 'r', encoding='utf-8') as f:
                                existing_data = json.load(f)
                            if 'all_unmatched_files' not in existing_data:
                                existing_data['all_unmatched_files'] = []
                            existing_data['all_unmatched_files'].extend(unmatched_log)
                            existing_data['last_update'] = datetime.now().isoformat()
                            existing_data['total_models_processed'] = models_processed_count
                        else:
                            existing_data = {
                                'started': datetime.now().isoformat(),
                                'command': 'rescan-linked',
                                'total_models_to_process': len(models_needing_associations),
                                'all_unmatched_files': unmatched_log,
                                'last_update': datetime.now().isoformat(),
                                'total_models_processed': models_processed_count
                            }
                        
                        with open(log_path, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)
                        
                        if self.verbose:
                            print(f"📋 Logged {len(unmatched_log)} unmatched files to {log_path}")
                        
                    except Exception as e:
                        print(f"⚠️ Error writing unmatched log: {e}")
                    
                    # Clear the batch
                    unmatched_log.clear()
            
            # Process models needing associations
            for model_file_id, model_sf_id, model_path, model_name in models_needing_associations:
                if self.verbose:
                    print(f"🔍 Processing model: {model_name}")
                    print(f"   Database path: {model_path}")
                
                # Resolve actual file location: check database path first, then target directory
                actual_model_path = self._resolve_model_path(model_path, model_name)
                
                if not actual_model_path:
                    if self.verbose:
                        print(f"   ⚠️ Model file not found in database path or destination directory")
                        print(f"      Database path checked: {model_path}")
                        dest_dir = self.scanner.config.get('destination_directory', 'Not configured')
                        print(f"      Destination directory searched: {dest_dir}")
                    continue
                elif actual_model_path != model_path:
                    if self.verbose:
                        print(f"   Resolved to: {actual_model_path}")
                else:
                    if self.verbose:
                        print(f"   Using database path (file exists)")
                
                # Use the resolved path for all operations
                working_model_path = actual_model_path
                
                model_dir = os.path.dirname(working_model_path)
                model_base_name = os.path.splitext(model_name)[0]
                
                # Look for associated files in the same directory (both database and resolved paths)
                cursor.execute('''
                    SELECT sf.id, sf.file_path, sf.file_name
                    FROM scanned_files sf
                    WHERE sf.file_path LIKE ? AND sf.file_type IN ('text', 'image', 'video', 'json', 'unknown')
                ''', (f"{model_dir}%",))
                
                potential_files = cursor.fetchall()
                same_dir_files = [f for f in potential_files if os.path.dirname(f[1]) == model_dir]
                
                # If we resolved to a different path, also check that directory
                if working_model_path != model_path:
                    actual_model_dir = os.path.dirname(working_model_path)
                    cursor.execute('''
                        SELECT sf.id, sf.file_path, sf.file_name
                        FROM scanned_files sf
                        WHERE sf.file_path LIKE ? AND sf.file_type IN ('text', 'image', 'video', 'json', 'unknown')
                    ''', (f"{actual_model_dir}%",))
                    
                    actual_potential_files = cursor.fetchall()
                    actual_same_dir_files = [f for f in actual_potential_files if os.path.dirname(f[1]) == actual_model_dir]
                    
                    # Combine both sets, removing duplicates by file_path
                    seen_paths = {f[1] for f in same_dir_files}
                    for f in actual_same_dir_files:
                        if f[1] not in seen_paths:
                            same_dir_files.append(f)
                            seen_paths.add(f[1])
                
                if self.verbose:
                    print(f"   Found {len(same_dir_files)} potential associated files in directory")
                
                # Use FileScanner's sophisticated correlation logic
                candidate_files = [assoc_path for _, assoc_path, _ in same_dir_files]
                
                # Get all associated files that should be linked to this model using server database
                try:
                    associated_files = temp_scanner.find_associated_files(working_model_path)
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️ Could not run advanced correlation (file access issue): {e}")
                    associated_files = []
                
                # Calculate model hashes for enhanced content correlation
                try:
                    model_sha256, model_autov3, _ = temp_scanner.calculate_hashes(working_model_path)
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️ Could not calculate model hashes: {e}")
                    model_sha256, model_autov3 = "", ""
                
                files_linked = 0
                
                # Helper functions for filename matching (available to all methods)
                def has_version_info(filename):
                    """Check if filename contains version information"""
                    import re
                    base = os.path.splitext(filename)[0]
                    if base.endswith('.metadata'):
                        base = base[:-9]
                    
                    # Look for various version patterns:
                    # - v1.0, v2.1 (standard semantic versioning)
                    # - V11, V23 (single version numbers)
                    # - V11_0.85_V8_0.15 (complex version combinations)
                    # - _e2, -e2, _v2, -v2 (epoch/version suffixes)
                    # - _2, -2 (simple number suffixes)
                    version_patterns = [
                        r'[v]\d+\.\d+',                     # v1.0, v2.1
                        r'[V]\d+(?:_[\d\.]+)*(?:_V\d+)*',   # V11, V11_0.85_V8_0.15
                        r'[_\-][v]\d+',                     # _v1, _v2, -v1, -v2
                        r'[_\-][e]\d+',                     # _e2, -e2 (epoch versions)
                        r'[_\-]\d+$',                       # _2, -2 (trailing numbers)
                    ]
                    
                    for pattern in version_patterns:
                        if re.search(pattern, base, re.IGNORECASE):
                            return True
                    return False
                
                def extract_adaptive_name(filename, use_version_aware=True):
                    """Extract filename - version-aware or base name depending on context"""
                    # Handle special compound extensions first
                    if filename.endswith('.civitai.info'):
                        base = filename[:-12]  # Remove .civitai.info
                    elif filename.endswith('.metadata.json'):
                        base = filename[:-14]  # Remove .metadata.json  
                    else:
                        base = os.path.splitext(filename)[0]
                        # Remove .metadata suffix
                        if base.endswith('.metadata'):
                            base = base[:-9]
                    
                    # Clean up any trailing dots
                    base = base.rstrip('.')
                    
                    if use_version_aware:
                        # Keep version info
                        return base
                    else:
                        # Remove version and hash info for base name matching
                        import re
                        cleaned = base
                        
                        # Enhanced base name extraction for better matching
                        cleaned = base
                        
                        # Remove common training/version patterns
                        cleaned = re.sub(r'[_\-][v]\d+\.\d+', '', cleaned, flags=re.IGNORECASE)     # _v1.0, -v2.1
                        cleaned = re.sub(r'[_\-][V]\d+(?:_[\d\.]+)*', '', cleaned)                # _V11, -V23
                        cleaned = re.sub(r'[_\-][v]\d+', '', cleaned, flags=re.IGNORECASE)        # _v1, -v2
                        cleaned = re.sub(r'[_\-][e]\d+', '', cleaned, flags=re.IGNORECASE)        # _e2, -e2
                        
                        # Remove training step patterns (common in model names)
                        cleaned = re.sub(r'[_\-]\d+[_\-]\d+$', '', cleaned)                       # -1000-128, _500_64
                        cleaned = re.sub(r'[_\-]\d{3,}$', '', cleaned)                            # -1000, _512 (3+ digits)
                        cleaned = re.sub(r'[_\-]\d+$', '', cleaned)                               # _2, -2 (any trailing number)
                        
                        # Remove long hashes and technical suffixes
                        cleaned = re.sub(r'[_\-]*\d{7,}[_\-]*', '', cleaned)                      # Long hashes
                        cleaned = re.sub(r'[_\-]+(ss|sample|preview|thumb)$', '', cleaned, flags=re.IGNORECASE)  # _ss, -sample suffixes
                        
                        # Clean up separators and whitespace
                        cleaned = re.sub(r'[_\-]+$', '', cleaned).strip()                         # Trailing separators
                        cleaned = re.sub(r'^[_\-]+', '', cleaned).strip()                         # Leading separators
                        
                        return cleaned
                
                for assoc_sf_id, assoc_path, assoc_name in same_dir_files:
                    assoc_base_name = os.path.splitext(assoc_name)[0]
                    
                    # Check if this file should be associated using sophisticated logic
                    should_link = False
                    match_reason = ""
                    
                    # METHOD 1: Check if FileScanner correlation identified this file
                    if assoc_path in associated_files:
                        should_link = True
                        match_reason = "Advanced correlation"
                        if self.verbose:
                            print(f"   ✓ Advanced correlation match: {assoc_name}")
                    
                    # METHOD 1.5: Enhanced content-based correlation (ALWAYS check for metadata files and media)
                    content_correlation_result = None
                    if assoc_name.lower().endswith(('.json', '.txt', '.md', '.yml', '.yaml', '.metadata.json', '.civitai.info', '.mp4', '.webp', '.png', '.jpg', '.jpeg')):
                        content_correlation_result = self._analyze_content_correlation(
                            assoc_path, assoc_name, model_name, model_sha256 or "", model_autov3 or ""
                        )
                        if content_correlation_result['should_link'] and not should_link:
                            should_link = True
                            match_reason = f"Content correlation ({content_correlation_result['reason']})"
                            if self.verbose:
                                print(f"   ✓ Content correlation match: {assoc_name}")
                                print(f"      Reason: {content_correlation_result['details']}")
                    
                    # METHOD 2: Adaptive filename matching - version-aware or base name
                    elif not should_link:
                        # Check if any files in this directory have version info
                        dir_files = [assoc_name for _, _, assoc_name in same_dir_files] + [model_name]
                        has_versions = any(has_version_info(f) for f in dir_files)
                        
                        # Try version-aware matching first if versions are detected
                        match_found = False
                        if has_versions:
                            # Try version-aware matching
                            model_match_name = extract_adaptive_name(model_name, use_version_aware=True)
                            assoc_match_name = extract_adaptive_name(assoc_name, use_version_aware=True)
                            
                            if (model_match_name.lower() == assoc_match_name.lower() and 
                                assoc_name.lower().endswith(('.json', '.txt', '.md', '.yml', '.yaml', '.metadata.json', '.civitai.info', '.mp4', '.webp', '.png', '.jpg', '.jpeg'))):
                                should_link = True
                                match_reason = "Version-aware filename match"
                                match_found = True
                                if self.verbose:
                                    print(f"   ✓ Version-aware filename match: {assoc_name}")
                                    print(f"      Matched: '{model_match_name}' == '{assoc_match_name}'")
                        
                        # If version-aware matching failed or no versions detected, try base name matching
                        if not match_found:
                            model_match_name = extract_adaptive_name(model_name, use_version_aware=False)
                            assoc_match_name = extract_adaptive_name(assoc_name, use_version_aware=False)
                            
                            if (model_match_name.lower() == assoc_match_name.lower() and 
                                assoc_name.lower().endswith(('.json', '.txt', '.md', '.yml', '.yaml', '.metadata.json', '.civitai.info', '.mp4', '.webp', '.png', '.jpg', '.jpeg'))):
                                should_link = True
                                match_reason = "Base name filename match"
                                if self.verbose:
                                    print(f"   ✓ Base name filename match: {assoc_name}")
                                    print(f"      Matched: '{model_match_name}' == '{assoc_match_name}' (after removing versions)")
                    
                    # METHOD 2.5: Base name match with target directory/database check
                    elif not should_link:
                        # Check if base names match - if so, search target directory and database
                        model_base_name_clean = extract_adaptive_name(model_name, use_version_aware=False)
                        assoc_base_name_clean = extract_adaptive_name(assoc_name, use_version_aware=False)
                        
                        if (model_base_name_clean.lower() == assoc_base_name_clean.lower() and 
                            assoc_name.lower().endswith(('.json', '.txt', '.md', '.yml', '.yaml', '.metadata.json', '.civitai.info', '.mp4', '.webp', '.png', '.jpg', '.jpeg'))):
                            
                            # Base names match! Now check target directory and database for the file
                            target_file_found = False
                            target_file_path = None
                            
                            # Check if file exists in target directory
                            target_dir = self.scanner.config.get('target_directory', '')
                            if target_dir:
                                import glob
                                
                                # Search for the file in target directory recursively
                                search_pattern = os.path.join(target_dir, "**", assoc_name)
                                matches = glob.glob(search_pattern, recursive=True)
                                if matches:
                                    target_file_path = matches[0]  # Use first match
                                    target_file_found = True
                                    if self.verbose:
                                        print(f"   🔍 Found in target directory: {target_file_path}")
                            
                            # Check database for file location if not found in target
                            if not target_file_found:
                                try:
                                    cursor.execute('''
                                        SELECT file_path FROM scanned_files 
                                        WHERE file_name = ? OR file_path LIKE ?
                                    ''', (assoc_name, f"%{assoc_name}"))
                                    db_result = cursor.fetchone()
                                    if db_result and os.path.exists(db_result[0]):
                                        target_file_path = db_result[0]
                                        target_file_found = True
                                        if self.verbose:
                                            print(f"   🔍 Found in database: {target_file_path}")
                                except Exception as e:
                                    if self.verbose:
                                        print(f"   ⚠️ Database search error: {e}")
                            
                            # If file found, use previously computed correlation or perform new analysis
                            if target_file_found and target_file_path:
                                # Use existing correlation result if available, otherwise analyze
                                if content_correlation_result is not None:
                                    correlation_result = content_correlation_result
                                elif model_sha256:
                                    correlation_result = self._analyze_content_correlation(
                                        target_file_path, assoc_name, model_name, model_sha256, model_autov3 or ""
                                    )
                                else:
                                    correlation_result = {'should_link': False, 'reason': 'no_hash', 'details': 'No model hash available'}
                                
                                if correlation_result['should_link']:
                                    should_link = True
                                    match_reason = f"Base name + target search + content correlation ({correlation_result['reason']})"
                                    if self.verbose:
                                        print(f"   ✓ Base name + target search + content correlation: {assoc_name}")
                                        print(f"      Found at: {target_file_path}")
                                        print(f"      Correlation: {correlation_result['details']}")
                                    # Update the assoc_path to point to the found file
                                    assoc_path = target_file_path
                                elif target_file_found:
                                    # Base name matches and file exists - check if we should link based on content quality
                                    confidence_score = correlation_result.get('confidence_score', 0)
                                    if confidence_score >= 30:  # Moderate confidence threshold
                                        should_link = True
                                        match_reason = f"Base name + target search + moderate correlation (score: {confidence_score})"
                                        if self.verbose:
                                            print(f"   ✓ Base name + target search + moderate correlation: {assoc_name}")
                                            print(f"      Found at: {target_file_path}")
                                            print(f"      Confidence: {confidence_score}% - {correlation_result['details']}")
                                        # Update the assoc_path to point to the found file
                                        assoc_path = target_file_path
                                    else:
                                        if self.verbose:
                                            print(f"   ⚠️ Base name matches but low content correlation: {assoc_name}")
                                            print(f"      Found at: {target_file_path}")
                                            print(f"      Low confidence: {confidence_score}% - {correlation_result['details']}")
                            
                            # Even if file not found in target, check if we have strong content correlation from current location
                            elif not target_file_found and content_correlation_result and content_correlation_result['should_link']:
                                should_link = True
                                match_reason = f"Base name + strong content correlation ({content_correlation_result['reason']})"
                                if self.verbose:
                                    print(f"   ✓ Base name + strong content correlation: {assoc_name}")
                                    print(f"      Correlation: {content_correlation_result['details']}")
                    
                    # METHOD 3: Enhanced exact name match (with content correlation support)
                    elif assoc_base_name == model_base_name:
                        # For documentation/metadata files with exact name matches, use content correlation to boost confidence
                        if (content_correlation_result and content_correlation_result.get('confidence_score', 0) >= 50) or assoc_name.lower().endswith(('.md', '.txt', '.json', '.yml', '.yaml', '.civitai.info')):
                            should_link = True
                            if content_correlation_result and content_correlation_result['should_link']:
                                match_reason = f"Exact name + content correlation ({content_correlation_result['reason']})"
                                if self.verbose:
                                    print(f"   ✓ Exact name + content correlation match: {assoc_name}")
                                    print(f"      Correlation: {content_correlation_result['details']}")
                            else:
                                match_reason = "Exact name match (documentation file)"
                                if self.verbose:
                                    print(f"   ✓ Exact name match (documentation): {assoc_name}")
                        else:
                            should_link = True
                            match_reason = "Exact name match"
                            if self.verbose:
                                print(f"   ✓ Exact name match: {assoc_name}")
                    
                    # METHOD 4: Single model scenario (fallback)
                    elif len(models_needing_associations) == 1:  # Only one model in this batch
                        # Check if only one model in this directory
                        cursor.execute('''
                            SELECT COUNT(*) FROM model_files mf
                            JOIN scanned_files sf ON mf.scanned_file_id = sf.id
                            WHERE sf.file_path LIKE ?
                        ''', (f"{model_dir}%",))
                        models_in_dir = cursor.fetchone()[0]
                        if models_in_dir == 1:
                            should_link = True
                            match_reason = "Single model scenario"
                            if self.verbose:
                                print(f"   ✓ Single model scenario: {assoc_name}")
                    
                    if should_link:
                        # Check if association already exists
                        cursor.execute('''
                            SELECT 1 FROM associated_files 
                            WHERE model_file_id = ? AND scanned_file_id = ?
                        ''', (model_file_id, assoc_sf_id))
                        
                        if not cursor.fetchone():
                            assoc_type = self._determine_association_type(assoc_path)
                            
                            # Try to write to database, handle read-only gracefully
                            try:
                                cursor.execute('''
                                    INSERT INTO associated_files 
                                    (model_file_id, scanned_file_id, association_type, source_path, is_moved)
                                    VALUES (?, ?, ?, ?, 0)
                                ''', (model_file_id, assoc_sf_id, assoc_type, assoc_path))
                                associations_created += 1
                                associations_since_last_commit += 1
                                files_linked += 1
                                if self.verbose:
                                    print(f"   🔗 Linked {assoc_type}: {assoc_name}")
                                
                                # Commit batch every 100 associations
                                if associations_since_last_commit >= COMMIT_BATCH_SIZE:
                                    commit_batch_associations()
                                    
                            except Exception as e:
                                if "readonly database" in str(e).lower():
                                    # Database is read-only, just log what would be linked
                                    associations_created += 1
                                    files_linked += 1
                                    if self.verbose:
                                        print(f"   🔗 Would link {assoc_type}: {assoc_name} (READ-ONLY MODE)")
                                else:
                                    raise  # Re-raise if it's a different error
                        else:
                            if self.verbose:
                                print(f"   ⚠️ Already linked: {assoc_name}")
                        
                        missing_associations_found += 1
                    else:
                        # Log unmatched file with detailed information
                        unmatched_entry = {
                            'model_name': model_name,
                            'model_path': model_path,
                            'model_base_name': model_base_name,
                            'unmatched_file': assoc_name,
                            'unmatched_path': assoc_path,
                            'unmatched_base_name': assoc_base_name,
                            'attempted_methods': [],
                            'reason': 'No correlation methods succeeded'
                        }
                        
                        # Document what was attempted
                        if assoc_path in associated_files:
                            unmatched_entry['attempted_methods'].append('Advanced correlation - FAILED')
                        else:
                            unmatched_entry['attempted_methods'].append('Advanced correlation - NOT FOUND')
                        
                        # Check adaptive filename matching attempt
                        def has_version_info_log(filename):
                            import re
                            base = os.path.splitext(filename)[0]
                            if base.endswith('.metadata'):
                                base = base[:-9]
                            
                            # Look for various version patterns (same as main function)
                            version_patterns = [
                                r'[v]\d+\.\d+',                     # v1.0, v2.1
                                r'[V]\d+(?:_[\d\.]+)*(?:_V\d+)*',   # V11, V11_0.85_V8_0.15
                                r'[_\-][v]\d+',                     # _v1, _v2, -v1, -v2
                                r'[_\-][e]\d+',                     # _e2, -e2 (epoch versions)
                                r'[_\-]\d+$',                       # _2, -2 (trailing numbers)
                            ]
                            
                            for pattern in version_patterns:
                                if re.search(pattern, base, re.IGNORECASE):
                                    return True
                            return False
                        
                        def extract_adaptive_name_log(filename, use_version_aware=True):
                            # Handle special compound extensions first
                            if filename.endswith('.civitai.info'):
                                base = filename[:-12]  # Remove .civitai.info
                            elif filename.endswith('.metadata.json'):
                                base = filename[:-14]  # Remove .metadata.json  
                            else:
                                base = os.path.splitext(filename)[0]
                                if base.endswith('.metadata'):
                                    base = base[:-9]
                            
                            # Clean up any trailing dots
                            base = base.rstrip('.')
                            
                            if use_version_aware:
                                return base
                            else:
                                import re
                                
                                # Enhanced base name extraction for better matching (same as main function)
                                cleaned = base
                                
                                # Remove common training/version patterns
                                cleaned = re.sub(r'[_\-][v]\d+\.\d+', '', cleaned, flags=re.IGNORECASE)     # _v1.0, -v2.1
                                cleaned = re.sub(r'[_\-][V]\d+(?:_[\d\.]+)*', '', cleaned)                # _V11, -V23
                                cleaned = re.sub(r'[_\-][v]\d+', '', cleaned, flags=re.IGNORECASE)        # _v1, -v2
                                cleaned = re.sub(r'[_\-][e]\d+', '', cleaned, flags=re.IGNORECASE)        # _e2, -e2
                                
                                # Remove training step patterns (common in model names)
                                cleaned = re.sub(r'[_\-]\d+[_\-]\d+$', '', cleaned)                       # -1000-128, _500_64
                                cleaned = re.sub(r'[_\-]\d{3,}$', '', cleaned)                            # -1000, _512 (3+ digits)
                                cleaned = re.sub(r'[_\-]\d+$', '', cleaned)                               # _2, -2 (any trailing number)
                                
                                # Remove long hashes and technical suffixes
                                cleaned = re.sub(r'[_\-]*\d{7,}[_\-]*', '', cleaned)                      # Long hashes
                                cleaned = re.sub(r'[_\-]+(ss|sample|preview|thumb)$', '', cleaned, flags=re.IGNORECASE)  # _ss, -sample suffixes
                                
                                # Clean up separators and whitespace
                                cleaned = re.sub(r'[_\-]+$', '', cleaned).strip()                         # Trailing separators
                                cleaned = re.sub(r'^[_\-]+', '', cleaned).strip()                         # Leading separators
                                
                                return cleaned
                        
                        # Check if directory has versions
                        dir_files = [f[2] for f in same_dir_files] + [model_name]
                        has_versions = any(has_version_info_log(f) for f in dir_files)
                        
                        # Try version-aware matching first if versions detected
                        version_matched = False
                        base_matched = False
                        
                        if has_versions:
                            # Try version-aware matching  
                            model_version_aware = extract_adaptive_name_log(model_name, use_version_aware=True)
                            assoc_version_aware = extract_adaptive_name_log(assoc_name, use_version_aware=True)
                            
                            if (model_version_aware.lower() == assoc_version_aware.lower() and 
                                assoc_name.lower().endswith(('.json', '.txt', '.md', '.yml', '.yaml', '.metadata.json', '.civitai.info', '.mp4', '.webp', '.png', '.jpg', '.jpeg'))):
                                version_matched = True
                                unmatched_entry['attempted_methods'].append(f'Version-aware filename match - MATCHED ("{model_version_aware}" == "{assoc_version_aware}") but failed other checks')
                            else:
                                unmatched_entry['attempted_methods'].append(f'Version-aware filename match - FAILED ("{model_version_aware}" != "{assoc_version_aware}")')
                        
                        # Try base name matching (always try this as fallback)
                        model_base = extract_adaptive_name_log(model_name, use_version_aware=False)
                        assoc_base = extract_adaptive_name_log(assoc_name, use_version_aware=False)
                        
                        if (model_base.lower() == assoc_base.lower() and 
                            assoc_name.lower().endswith(('.json', '.txt', '.md', '.yml', '.yaml', '.metadata.json', '.civitai.info', '.mp4', '.webp', '.png', '.jpg', '.jpeg'))):
                            base_matched = True
                            if not version_matched:  # Only log if version matching didn't already succeed
                                unmatched_entry['attempted_methods'].append(f'Base name filename match - MATCHED ("{model_base}" == "{assoc_base}") but failed other checks')
                        else:
                            if not has_versions:  # Only log base name failure if no version matching was attempted
                                if model_base.lower() == assoc_base.lower():
                                    unmatched_entry['attempted_methods'].append(f'Base name filename match - MATCHED ("{model_base}" == "{assoc_base}") but extension not allowed')
                                else:
                                    unmatched_entry['attempted_methods'].append(f'Base name filename match - FAILED ("{model_base}" != "{assoc_base}")')
                        
                        if assoc_base_name == model_base_name:
                            unmatched_entry['attempted_methods'].append('Exact name match - MATCHED but failed other checks')
                        else:
                            unmatched_entry['attempted_methods'].append(f'Exact name match - FAILED ("{assoc_base_name}" != "{model_base_name}")')
                        
                        unmatched_entry['attempted_methods'].append(f'Single model scenario - {"PASSED" if len(models_needing_associations) == 1 else "FAILED (multiple models)"}')
                        
                        unmatched_log.append(unmatched_entry)
                        
                        if self.verbose:
                            print(f"   ❌ No match: {assoc_name}")
                            print(f"      Attempted: {', '.join(unmatched_entry['attempted_methods'])}")
                
                if self.verbose:
                    print(f"   📊 Result: {files_linked} new associations created for {model_name}")
                    print("")
                
                # Increment model counter and write log every 5 models
                models_processed_count += 1
                if models_processed_count % 5 == 0:
                    write_unmatched_log_batch()
                    if self.verbose:
                        print(f"📋 Processed {models_processed_count}/{len(models_needing_associations)} models, log updated")
            
            # Write any remaining unmatched files
            if unmatched_log:
                write_unmatched_log_batch()
            
            # Verify existing associations are still valid
            cursor.execute('''
                SELECT af.id, af.model_file_id, af.scanned_file_id, af.source_path
                FROM associated_files af
                JOIN model_files mf ON af.model_file_id = mf.id
                JOIN scanned_files sf ON af.scanned_file_id = sf.id
            ''')
            
            existing_associations = cursor.fetchall()
            
            if self.verbose:
                print(f"🔍 Verifying {len(existing_associations)} existing associations...")
            
            invalid_associations = 0
            
            for assoc_id, model_file_id, scanned_file_id, source_path in existing_associations:
                # Check if the file still exists
                if not os.path.exists(source_path):
                    cursor.execute('DELETE FROM associated_files WHERE id = ?', (assoc_id,))
                    invalid_associations += 1
                    if self.verbose:
                        print(f"   ❌ Removed invalid association: {os.path.basename(source_path)}")
                        print(f"      (File no longer exists at: {source_path})")
                elif self.verbose:
                    print(f"   ✓ Valid association: {os.path.basename(source_path)}")
            
            if self.verbose and invalid_associations > 0:
                print(f"📊 Removed {invalid_associations} invalid associations")
            elif self.verbose:
                print(f"📊 All existing associations are valid")
            
            # Commit any remaining associations before final commit
            commit_batch_associations()
            
            # Try to commit changes, handle read-only gracefully
            try:
                conn.commit()
            except Exception as e:
                if "readonly database" in str(e).lower():
                    if self.verbose:
                        print("   ℹ️ Database is read-only - running in analysis mode")
                else:
                    raise  # Re-raise if it's a different error
            
            self.print_progress(f"Verification complete:")
            self.print_progress(f"  - Models checked: {len(models_needing_associations):,}")
            self.print_progress(f"  - Missing associations found: {missing_associations_found:,}")
            self.print_progress(f"  - New associations created: {associations_created:,}")
            self.print_progress(f"  - Invalid associations removed: {invalid_associations:,}")
            self.print_progress(f"  - Orphaned files detected: {len(orphaned_files):,}")
            
            if associations_created > 0:
                self.print_progress(f"✓ Fixed {associations_created} missing associations")
            if invalid_associations > 0:
                self.print_progress(f"✓ Cleaned up {invalid_associations} invalid associations")
            if associations_created == 0 and invalid_associations == 0:
                self.print_progress("✓ All associations are correct, no changes needed")
            
            # Final log summary
            if os.path.exists(log_path):
                try:
                    import json
                    with open(log_path, 'r', encoding='utf-8') as f:
                        final_log_data = json.load(f)
                    total_unmatched = len(final_log_data.get('all_unmatched_files', []))
                    self.print_progress(f"📋 Final unmatched files log: {log_path}")
                    self.print_progress(f"   Total unmatched files: {total_unmatched}")
                    if self.verbose:
                        print(f"\n📋 Complete Unmatched Files Analysis:")
                        print(f"   Log file: {log_path}")
                        print(f"   Total models processed: {models_processed_count}")
                        print(f"   Total unmatched files: {total_unmatched}")
                        print(f"   Review the JSON file for detailed analysis")
                except Exception as e:
                    self.print_progress(f"⚠️ Error reading final log summary: {e}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in verify linked files: {e}")
            self.stats['errors'] += 1
            return False
    
    def step_rebuild_all_links(self) -> bool:
        """Step: Completely rebuild all file associations from scratch"""
        try:
            self.print_step("REBUILD ALL LINKS", 
                           "Completely rebuilding all file associations from scratch")
            
            # Initialize scanner to access database
            if not hasattr(self, 'scanner'):
                from file_scanner import FileScanner
                self.scanner = FileScanner(self.config_path)
            
            conn = self.scanner.db.conn
            cursor = conn.cursor()
            
            # Get model extensions from FileScanner to ensure we cover all types
            model_extensions = list(self.scanner.MODEL_EXTENSIONS)
            extensions_tuple = "(" + ",".join(f"'{ext}'" for ext in model_extensions) + ")"
            
            # Get all model files from scanned_files (not just those with extracted metadata)
            cursor.execute(f'''
                SELECT sf.id, sf.file_name, sf.file_path
                FROM scanned_files sf
                WHERE sf.file_type = 'model' AND sf.extension IN {extensions_tuple}
                ORDER BY sf.file_path
            ''')
            
            model_files = cursor.fetchall()
            self.print_progress(f"Found {len(model_files)} model files to process")
            
            # Get all non-model files that could be associated
            cursor.execute('''
                SELECT sf.id, sf.file_path, sf.file_name
                FROM scanned_files sf
                WHERE sf.file_type IN ('text', 'image', 'video', 'json', 'unknown')
                ORDER BY sf.file_path
            ''')
            
            potential_associated_files = cursor.fetchall()
            self.print_progress(f"Found {len(potential_associated_files)} potential associated files")
            
            # Clear existing associations to rebuild them
            cursor.execute('DELETE FROM associated_files')
            self.print_progress("Cleared all existing file associations")
            
            # Clear temporary associations too
            cursor.execute('DROP TABLE IF EXISTS temp_file_associations')
            cursor.execute('DROP TABLE IF EXISTS temp_associations')
            
            # Create a table to track model->associated file relationships
            cursor.execute('''
                CREATE TEMPORARY TABLE temp_associations (
                    model_scanned_file_id INTEGER,
                    assoc_scanned_file_id INTEGER,
                    association_type TEXT,
                    source_path TEXT
                )
            ''')
            
            linked_count = 0
            processed_count = 0
            
            # Process each model
            for model_sf_id, model_file_name, model_path in model_files:
                processed_count += 1
                if processed_count % 1000 == 0:
                    self.print_progress(f"Processed {processed_count:,}/{len(model_files):,} models...")
                
                # Extract directory and base name from model path
                model_dir = os.path.dirname(model_path)
                model_base_name = os.path.splitext(model_file_name)[0]
                
                # Find associated files for this model
                associated_files = []
                
                # 1. Same directory, exact name match
                for sf_id, sf_path, sf_name in potential_associated_files:
                    sf_dir = os.path.dirname(sf_path)
                    if sf_dir != model_dir:
                        continue
                    
                    sf_base_name = os.path.splitext(sf_name)[0]
                    
                    # Exact name match
                    if sf_base_name == model_base_name:
                        assoc_type = self._determine_association_type(sf_path)
                        associated_files.append((sf_id, sf_path, assoc_type))
                
                # 2. If only one model in directory, link all files in that directory
                same_dir_models = [m for m in model_files if os.path.dirname(m[2]) == model_dir]
                if len(same_dir_models) == 1:
                    for sf_id, sf_path, sf_name in potential_associated_files:
                        sf_dir = os.path.dirname(sf_path)
                        if sf_dir == model_dir:
                            sf_base_name = os.path.splitext(sf_name)[0]
                            # Avoid duplicates from exact matches above
                            if sf_base_name != model_base_name:
                                assoc_type = self._determine_association_type(sf_path)
                                associated_files.append((sf_id, sf_path, assoc_type))
                
                # Insert associations into temporary table
                for sf_id, sf_path, assoc_type in associated_files:
                    cursor.execute('''
                        INSERT INTO temp_associations 
                        (model_scanned_file_id, assoc_scanned_file_id, association_type, source_path)
                        VALUES (?, ?, ?, ?)
                    ''', (model_sf_id, sf_id, assoc_type, sf_path))
                    linked_count += 1
            
            # Now insert into associated_files table for models that exist in model_files
            cursor.execute('''
                INSERT INTO associated_files 
                (model_file_id, scanned_file_id, association_type, source_path, is_moved)
                SELECT mf.id, ta.assoc_scanned_file_id, ta.association_type, ta.source_path, 0
                FROM temp_associations ta
                JOIN model_files mf ON mf.scanned_file_id = ta.model_scanned_file_id
            ''')
            
            associated_with_metadata = cursor.rowcount
            
            conn.commit()
            
            self.print_progress(f"Complete rebuild finished:")
            self.print_progress(f"  - Models processed: {processed_count:,}")
            self.print_progress(f"  - Total associations found: {linked_count:,}")
            self.print_progress(f"  - Associations linked to extracted metadata: {associated_with_metadata:,}")
            if processed_count > 0:
                self.print_progress(f"  - Average associations per model: {linked_count/processed_count:.1f}")
            
            if associated_with_metadata < linked_count:
                self.print_progress(f"  - Note: {linked_count - associated_with_metadata:,} associations are pending metadata extraction")
            
            return True
            
        except Exception as e:
            print(f"ERROR in rebuild all links: {e}")
            self.stats['errors'] += 1
            return False

    def step_rescan_and_repair_metadata_text(self) -> bool:
        """Step: Rescan and repair metadata/text file paths with intelligent path correction"""
        try:
            self.print_step("RESCAN AND REPAIR METADATA/TEXT FILE PATHS", 
                           "Intelligently scan filesystem and repair database path mismatches")
            
            # Initialize scanner to access database
            if not hasattr(self, 'scanner'):
                from file_scanner import FileScanner
                self.scanner = FileScanner(self.config_path)
            
            results = self.scanner.rescan_and_repair_text_file_paths()
            
            # Display results
            self.print_progress(f"Files scanned: {results.get('files_scanned', 0):,}")
            self.print_progress(f"Paths checked: {results.get('paths_checked', 0):,}")
            self.print_progress(f"Paths corrected: {results.get('paths_corrected', 0):,}")
            self.print_progress(f"Database updates: {results.get('database_updates', 0):,}")
            
            if results.get('errors', 0) > 0:
                print(f"⚠️  Errors encountered: {results.get('errors', 0)}")
                self.stats['errors'] += results.get('errors', 0)
            
            return True
            
        except Exception as e:
            print(f"ERROR in rescan and repair metadata/text: {e}")
            self.stats['errors'] += 1
            return False

    def step_migrate_tables(self) -> bool:
        """Migrate model files from scanned_files to model_files table"""
        try:
            self.print_step("MIGRATE TABLES", 
                          "Migrating model files from scanned_files to model_files table")
            
            import sqlite3
            
            # Connect to the server database using config
            server_db_path = self.get_server_database_path()
            if not os.path.exists(server_db_path):
                print(f"❌ Server database not found: {server_db_path}")
                return False
            
            conn = sqlite3.connect(server_db_path, timeout=30.0)
            cursor = conn.cursor()
            
            # Get all model files from scanned_files that don't have entries in model_files
            self.print_progress("Finding model files to migrate...")
            cursor.execute('''
                SELECT sf.id, sf.file_path, sf.file_name, sf.file_size, sf.sha256, sf.blake3
                FROM scanned_files sf
                LEFT JOIN model_files mf ON sf.id = mf.scanned_file_id
                WHERE sf.file_type = 'model' AND mf.id IS NULL
                ORDER BY sf.file_path
            ''')
            
            models_to_migrate = cursor.fetchall()
            total_models = len(models_to_migrate)
            
            if total_models == 0:
                self.print_progress("✅ No models need migration - all model files already in model_files table")
                conn.close()
                return True
            
            self.print_progress(f"Found {total_models} model files to migrate")
            
            # Migrate in batches for better performance
            batch_size = 1000
            migrated_count = 0
            
            for i in range(0, total_models, batch_size):
                batch = models_to_migrate[i:i + batch_size]
                batch_end = min(i + batch_size, total_models)
                
                self.print_progress(f"Migrating batch {i//batch_size + 1}: models {i+1}-{batch_end} of {total_models}")
                
                # Prepare batch insert data
                insert_data = []
                for scanned_file_id, file_path, file_name, file_size, sha256, blake3 in batch:
                    # Extract basic model info from filename
                    model_name = os.path.splitext(file_name)[0]
                    
                    # Determine model type using comprehensive extension and path analysis
                    model_type = self.determine_model_type_from_extension(file_path)
                    
                    insert_data.append((
                        scanned_file_id,
                        model_name,
                        'Unknown',  # base_model - will be enhanced later
                        model_type,
                        None,  # civitai_id
                        None,  # version_id
                        file_path,  # source_path
                        None,  # target_path
                        0,  # is_duplicate
                        None,  # duplicate_group_id
                        None,  # metadata_json
                        0,  # has_civitai_info
                        0,  # has_metadata_json
                        'pending',  # status
                        None  # error_message
                    ))
                
                # Batch insert into model_files
                cursor.executemany('''
                    INSERT INTO model_files 
                    (scanned_file_id, model_name, base_model, model_type, civitai_id, version_id, 
                     source_path, target_path, is_duplicate, duplicate_group_id, metadata_json, 
                     has_civitai_info, has_metadata_json, status, error_message, 
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            strftime('%s', 'now'), strftime('%s', 'now'))
                ''', insert_data)
                
                migrated_count += len(batch)
                
                # Commit this batch
                conn.commit()
                
                if migrated_count % 5000 == 0:
                    self.print_progress(f"  ✅ Migrated {migrated_count} models so far...")
            
            conn.close()
            
            self.print_progress(f"✅ Migration complete!")
            self.print_progress(f"Migrated {migrated_count} model files from scanned_files to model_files")
            self.print_progress(f"Note: Models migrated with basic info - run --step metadata to enhance with full metadata")
            
            self.stats['models_migrated'] = migrated_count
            
            return True
            
        except Exception as e:
            print(f"ERROR in table migration: {e}")
            import traceback
            traceback.print_exc()
            self.stats['errors'] += 1
            return False
    
    def _determine_association_type(self, file_path: str) -> str:
        """Determine the type of association based on file extension and name"""
        file_name = os.path.basename(file_path).lower()
        
        if file_name.endswith('.civitai.info'):
            return 'civitai_info'
        elif file_name.endswith('.metadata.json'):
            return 'metadata'
        elif any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
            return 'image'
        elif any(file_name.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
            return 'video'
        elif file_name.endswith('.txt'):
            return 'text'
        else:
            return 'other'
    
    def generate_final_report(self):
        """Generate final processing report"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total processing time: {duration:.2f} seconds")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        
        print(f"\nSummary:")
        print(f"  Files scanned: {self.stats['files_scanned']}")
        print(f"  Model files found: {self.stats['models_found']}")
        print(f"  Duplicate files found: {self.stats['duplicates_found']}")
        print(f"  Models sorted: {self.stats['models_sorted']}")
        print(f"  Civitai.info files generated: {self.stats['civitai_files_generated']}")
        print(f"  Errors encountered: {self.stats['errors']}")
        
        # Stop console capture and get output
        captured_output = self.console_capture.stop_capture()
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'dry_run': self.dry_run,
            'config_file': self.config_path,
            'stats': self.stats,
            'console_output': captured_output
        }
        
        # Create processing_report directory if it doesn't exist
        report_dir = "processing_report"
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(report_dir, report_file)
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\nDetailed report saved to: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save report file: {e}")
    
    def run_full_workflow(self) -> bool:
        """Run the complete model sorting workflow"""
        try:
            # Start capturing console output
            self.console_capture.start_capture()
            
            self.print_progress("Starting Model Sorter workflow...")
            self.print_progress(f"Configuration: {self.config_path}")
            self.print_progress(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE PROCESSING'}")
            
            # Step 1: Scan files
            if not self.step_1_scan_files():
                print("FAILED: File scanning failed")
                return False
            
            # Step 2: Extract metadata
            if not self.step_2_extract_metadata():
                print("FAILED: Metadata extraction failed")
                return False
            
            # Step 3: Detect duplicates
            if not self.step_3_detect_duplicates():
                print("FAILED: Duplicate detection failed")
                return False
            
            # Step 4: Sort models
            if not self.step_4_sort_models():
                print("FAILED: Model sorting failed")
                return False
            
            # Step 5: Generate civitai.info files
            if not self.step_5_generate_civitai_files():
                print("FAILED: Civitai.info generation failed")
                return False
            
            # Generate final report
            self.generate_final_report()
            
            return True
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            return False
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            self.stats['errors'] += 1
            return False
    
    def run_individual_step(self, step_name: str) -> bool:
        """Run an individual step of the workflow"""
        steps = {
            'scan': self.step_1_scan_files,
            'metadata': self.step_2_extract_metadata,
            'duplicates': self.step_3_detect_duplicates,
            'sort': self.step_4_sort_models,
            'civitai': self.step_5_generate_civitai_files,
            'rescan': self.step_rescan_metadata,
            'extract-metadata': self.step_extract_civitai_metadata,
            'retry-failed': self.step_retry_failed_metadata,
            'migrate-paths': None,  # Special handling needed
            'migrate-tables': self.step_migrate_tables,
            'recover-missing': self.step_recover_missing_files,
            'force-blake3': self.step_force_blake3_rescan,
            'update-tables': self.step_update_tables,
            'rescan-linked': self.step_rescan_linked_files,
            'rescan-linked-rebuild': self.step_rebuild_all_links,
            'rescan-and-repair-metadata-text': self.step_rescan_and_repair_metadata_text
        }
        
        if step_name not in steps:
            print(f"Unknown step: {step_name}")
            print(f"Available steps: {', '.join(steps.keys())}")
            return False
        
        # Special handling for migrate-paths step which requires parameters
        if step_name == 'migrate-paths':
            # This should be called from main() with proper arguments
            raise ValueError("migrate-paths step requires special handling with arguments")
        
        return steps[step_name]()
    
    def _analyze_content_correlation(self, file_path: str, file_name: str, model_name: str, model_sha256: str, model_autov3: str) -> dict:
        """Analyze file content for correlation with model beyond filename matching"""
        result = {
            'should_link': False,
            'reason': '',
            'details': '',
            'confidence_score': 0
        }
        
        try:
            # Skip if file doesn't exist or isn't accessible
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return result
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # PRIORITY CHECK: Exact base filename match for documentation files
            model_base = os.path.splitext(model_name)[0].lower()
            file_base = os.path.splitext(file_name)[0].lower()
            
            if (file_base == model_base and 
                file_ext in ['.md', '.txt', '.json', '.yml', '.yaml', '.civitai.info']):
                result.update({
                    'should_link': True,
                    'reason': 'exact_filename_match',
                    'details': f'Exact filename match: {file_base} == {model_base}',
                    'confidence_score': 95
                })
                return result
            
            # METHOD 1: Deep image metadata analysis
            if file_ext in ['.png', '.jpg', '.jpeg', '.webp']:
                try:
                    from file_scanner import extract_image_metadata, extract_comprehensive_metadata
                    
                    image_metadata = extract_image_metadata(file_path)
                    if image_metadata and image_metadata.get('ai_metadata'):
                        comprehensive_metadata = extract_comprehensive_metadata(file_path, image_metadata)
                        
                        # Check for exact hash matches in image metadata
                        if comprehensive_metadata.get('model_hash'):
                            img_hash = comprehensive_metadata['model_hash'].upper()
                            if (img_hash == model_sha256.upper() or 
                                (model_autov3 and img_hash == model_autov3.upper())):
                                result.update({
                                    'should_link': True,
                                    'reason': 'hash_match',
                                    'details': f'Model hash in image metadata matches: {img_hash[:10]}...',
                                    'confidence_score': 100
                                })
                                return result
                        
                        # Check model name in image metadata
                        if comprehensive_metadata.get('model_name'):
                            img_model = comprehensive_metadata['model_name'].lower()
                            model_clean = model_name.lower()
                            
                            # Exact model name match
                            if model_clean == img_model:
                                result.update({
                                    'should_link': True,
                                    'reason': 'exact_model_name',
                                    'details': f'Exact model name match in metadata: {img_model}',
                                    'confidence_score': 90
                                })
                                return result
                            
                            # Partial model name match (high confidence)
                            elif model_clean in img_model or img_model in model_clean:
                                result.update({
                                    'should_link': True,
                                    'reason': 'partial_model_name',
                                    'details': f'Model name similarity in metadata: {img_model}',
                                    'confidence_score': 75
                                })
                                return result
                        
                        # Check for significant parameter overlap
                        if comprehensive_metadata.get('raw_parameters'):
                            params_text = comprehensive_metadata['raw_parameters'].lower()
                            model_words = set(w for w in model_name.lower().replace('_', ' ').replace('-', ' ').split() if len(w) > 2)
                            matches = sum(1 for word in model_words if word in params_text)
                            
                            if matches >= 2 and len(model_words) > 0:
                                confidence = min(60 + (matches * 10), 85)
                                result.update({
                                    'should_link': True,
                                    'reason': 'parameter_correlation',
                                    'details': f'Model keywords found in generation parameters ({matches}/{len(model_words)} words)',
                                    'confidence_score': confidence
                                })
                                return result
                        
                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: Error analyzing image metadata for {file_name}: {e}")
            
            # METHOD 2: Enhanced text/metadata file analysis
            elif file_ext in ['.json', '.txt', '.md', '.yml', '.yaml', '.metadata.json', '.civitai.info']:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    content_lower = content.lower()
                    
                    # Direct hash reference (highest confidence) - only if hash available
                    if model_sha256 and (model_sha256.lower() in content_lower or (model_autov3 and model_autov3.lower() in content_lower)):
                        result.update({
                            'should_link': True,
                            'reason': 'direct_hash_reference',
                            'details': 'Model hash found directly in file content',
                            'confidence_score': 95
                        })
                        return result
                    
                    # Parse structured JSON for deeper analysis
                    if file_ext == '.json':
                        try:
                            json_data = json.loads(content)
                            
                            # Comprehensive field analysis
                            model_fields = ['model_name', 'name', 'title', 'filename', 'base_model']
                            hash_fields = ['sha256', 'model_hash', 'hash', 'hashes']
                            
                            # Check structured hash fields (only if hashes available)
                            if model_sha256:
                                for field in hash_fields:
                                    if field in json_data:
                                        field_value = str(json_data[field]).lower()
                                        if model_sha256.lower() in field_value or (model_autov3 and model_autov3.lower() in field_value):
                                            result.update({
                                                'should_link': True,
                                                'reason': f'structured_hash_{field}',
                                                'details': f'Hash match in JSON field "{field}"',
                                                'confidence_score': 95
                                            })
                                            return result
                            
                            # Check model identifier fields with enhanced matching
                            import re
                            model_clean = model_name.lower()
                            
                            # Create base name versions for comparison (remove version suffixes)
                            model_base = re.sub(r'[_\-][vVeE]\d+', '', model_clean)
                            model_base = re.sub(r'[_\-]\d+$', '', model_base)
                            
                            for field in model_fields:
                                if field in json_data:
                                    field_value = str(json_data[field]).lower()
                                    field_base = re.sub(r'[_\-][vVeE]\d+', '', field_value)
                                    field_base = re.sub(r'[_\-]\d+$', '', field_base)
                                    
                                    # Exact match
                                    if model_clean == field_value:
                                        result.update({
                                            'should_link': True,
                                            'reason': f'exact_{field}_match',
                                            'details': f'Exact match in JSON field "{field}": {field_value}',
                                            'confidence_score': 85
                                        })
                                        return result
                                    
                                    # Base name match (same model, different versions)
                                    elif model_base == field_base and len(model_base) > 3:
                                        result.update({
                                            'should_link': True,
                                            'reason': f'base_{field}_match',
                                            'details': f'Base name match in JSON field "{field}": {field_value} (base: {field_base})',
                                            'confidence_score': 80
                                        })
                                        return result
                                    
                                    # Substring match
                                    elif model_clean in field_value or field_value in model_clean:
                                        result.update({
                                            'should_link': True,
                                            'reason': f'partial_{field}_match',
                                            'details': f'Partial match in JSON field "{field}": {field_value}',
                                            'confidence_score': 70
                                        })
                                        return result
                            
                        except json.JSONDecodeError:
                            pass
                    
                    # Enhanced text analysis for model name correlation
                    model_words = set(w for w in model_name.lower().replace('_', ' ').replace('-', ' ').split() if len(w) > 2)
                    
                    # Also check for base model name (without version suffixes)
                    import re
                    model_base_name = re.sub(r'[_\-][vVeE]\d+', '', model_name.lower())
                    model_base_name = re.sub(r'[_\-]\d+$', '', model_base_name)
                    
                    # Direct base name match in text
                    if len(model_base_name) > 3 and model_base_name in content_lower:
                        result.update({
                            'should_link': True,
                            'reason': 'direct_name_in_text',
                            'details': f'Model base name "{model_base_name}" found directly in text content',
                            'confidence_score': 75
                        })
                        return result
                    
                    # Word-based correlation
                    if model_words:
                        matches = sum(1 for word in model_words if word in content_lower)
                        match_ratio = matches / len(model_words)
                        
                        # More lenient matching for text files
                        if matches >= 2 or (matches >= 1 and match_ratio >= 0.5 and len(model_words) <= 2):
                            confidence = min(50 + (matches * 8), 75)
                            result.update({
                                'should_link': True,
                                'reason': 'text_correlation',
                                'details': f'Text correlation ({matches}/{len(model_words)} significant words match)',
                                'confidence_score': confidence
                            })
                            return result
                        
                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: Error analyzing text file {file_name}: {e}")
            
            # METHOD 3: Filename pattern analysis for numeric IDs (Civitai correlation)
            if any(char.isdigit() for char in file_name[:10]):
                import re
                # Look for substantial numeric patterns that might be Civitai IDs
                numeric_patterns = re.findall(r'\d{5,}', file_name)
                if numeric_patterns:
                    # This could be enhanced with actual Civitai database lookup
                    result.update({
                        'should_link': False,  # Conservative - would need actual DB correlation
                        'reason': 'potential_id_correlation',
                        'details': f'Numeric pattern detected: {numeric_patterns[0]}',
                        'confidence_score': 30
                    })
        
        except Exception as e:
            if self.verbose:
                print(f"    Error in content correlation analysis for {file_name}: {e}")
        
        return result
    
    def _resolve_model_path(self, database_path: str, model_name: str) -> str:
        """
        Resolve the actual location of a model file.
        First checks database path, then searches target directory structure.
        Returns the actual path if found, empty string if not found.
        """
        # First, check if file exists at the database path
        if os.path.exists(database_path) and os.path.isfile(database_path):
            if self.verbose:
                print(f"      File exists at database path")
            return database_path
        
        if self.verbose:
            print(f"      Database path not accessible, searching target directory...")
        
        # If not found, search in target directory structure
        try:
            target_directory = self.scanner.config.get('destination_directory', '')
            if not target_directory:
                if self.verbose:
                    print(f"      No destination directory configured in config.ini")
                return ""
            
            if not os.path.exists(target_directory):
                if self.verbose:
                    print(f"      Destination directory does not exist: {target_directory}")
                return ""
            
            if self.verbose:
                print(f"      Searching destination directory: {target_directory}")
            
            # Search for the model file in the target directory structure
            model_filename = os.path.basename(database_path)
            
            if self.verbose:
                print(f"      Looking for filename: {model_filename}")
            
            # Try direct path first (fastest)
            direct_path = os.path.join(target_directory, model_filename)
            if os.path.exists(direct_path) and os.path.isfile(direct_path):
                if self.verbose:
                    print(f"      Found at direct path: {direct_path}")
                return direct_path
            
            # Common subdirectory glob patterns to search
            search_patterns = [
                # Model type subdirectories
                "loras/**/" + model_filename,
                "checkpoints/**/" + model_filename,
                "vaes/**/" + model_filename,
                "embeddings/**/" + model_filename,
                "controlnets/**/" + model_filename,
                "upscalers/**/" + model_filename,
                
                # Nested base model patterns
                "loras/*/**/" + model_filename,
                "checkpoints/*/**/" + model_filename,
                
                # Alternative structure patterns  
                "models/**/" + model_filename,
                "**/" + model_filename
            ]
            
            from pathlib import Path
            
            # Search using glob patterns
            target_path = Path(target_directory)
            
            for pattern in search_patterns:
                if self.verbose:
                    print(f"      Trying pattern: {pattern}")
                
                try:
                    # Use recursive glob
                    matches = list(target_path.glob(pattern))
                    for match in matches:
                        if match.is_file() and match.name == model_filename:
                            resolved_path = str(match)
                            if self.verbose:
                                print(f"      ✅ Found at: {resolved_path}")
                            return resolved_path
                except Exception as e:
                    if self.verbose:
                        print(f"      Warning: Error with pattern {pattern}: {e}")
                    continue
            
            # If still not found, try a broader search by filename only (slower but comprehensive)
            if self.verbose:
                print(f"      Pattern search failed, trying recursive filename search...")
                
            for root, dirs, files in os.walk(target_directory):
                if model_filename in files:
                    found_path = os.path.join(root, model_filename)
                    if self.verbose:
                        print(f"      ✅ Found via recursive search: {found_path}")
                    return found_path
        
        except Exception as e:
            if self.verbose:
                print(f"      Error searching target directory: {e}")
        
        # Return empty string if not found anywhere
        if self.verbose:
            print(f"      ❌ File not found anywhere")
        return ""
    
    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'scanner'):
                self.scanner.close()
            if hasattr(self, 'metadata_extractor'):
                self.metadata_extractor.close()
            if hasattr(self, 'duplicate_detector'):
                self.duplicate_detector.close()
            if hasattr(self, 'model_sorter'):
                self.model_sorter.close()
            if hasattr(self, 'civitai_generator'):
                self.civitai_generator.close()
        except Exception as e:
            print(f"Warning during cleanup: {e}")


def main():
    """Main entry point"""
    # Handle broken pipe errors (e.g., when output is piped to head)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    
    parser = argparse.ArgumentParser(
        description='🚀 Stable Diffusion Model Organizer - Complete Pipeline for Model Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📋 DETAILED COMMAND DESCRIPTIONS & COMBINATIONS:

CORE WORKFLOW STEPS (use with --step):
  scan          Scan directories and calculate hashes for model files
                • Combines with: --force-rescan, --verbose, --folder-limit
                • Purpose: Initial discovery of new models and hash calculation
                • Output: Populates database with file paths and SHA256/BLAKE3/AutoV3 hashes
                
  metadata      Extract metadata from SafeTensors, .metadata.json, .civitai.info
                • Combines with: --extract-metadata-limit, --extract-metadata-all, --verbose
                • Purpose: Gather model information for intelligent organization
                • Output: Model names, types, base models, descriptions in database
                
  duplicates    Identify duplicate models using SHA256 hash comparison  
                • Combines with: --verbose, --dry-run
                • Purpose: Find identical models with different names/locations
                • Output: Duplicate analysis report with recommended actions
                
  sort          Organize models into loras/base_model/model_name/ structure
                • Combines with: --dry-run (REQUIRED for safety), --verbose
                • Purpose: Physical file organization and directory creation
                • Output: Structured model hierarchy with associated files
                
  civitai       Generate missing .civitai.info files from metadata
                • Combines with: --verbose, --dry-run
                • Purpose: Create standardized model information files
                • Output: Properly formatted .civitai.info files for each model

🔄 SPECIALIZED OPERATIONS (use with --step or as standalone flags):
  rescan-linked         Verify and repair file associations (incremental check)
  rescan-linked-rebuild Complete rebuild of all file associations (nuclear option)
  extract-metadata      Enhanced metadata extraction with configurable limits
  retry-failed          Retry previously failed metadata extractions  
  migrate-paths         Update database paths after directory moves
  recover-missing       Find and recover missing model files
  force-blake3          Add BLAKE3 hashes to non-SafeTensors files
  update-tables         Ensure database completeness and fix missing entries

📊 METADATA EXTRACTION MODES (auto-selects extract-metadata step):
  --extract-metadata-limit N    Process exactly N unprocessed files
  --extract-metadata-all        Process all files missing metadata
  --extract-metadata-all-rescan Reprocess ALL files (including completed ones)

🎯 ASSOCIATION MANAGEMENT:
  --rescan-linked              ✅ RECOMMENDED: Quick verification and repair
                               • Fast incremental check of file relationships
                               • Fixes broken links and missing associations
                               
  --rescan-linked-rebuild      Complete association rebuild from scratch  
                               • Use only if database associations are completely broken
                               • Recreates all model↔associated file relationships
                               
  --rescan-and-repair-metadata-text Intelligent metadata path repair
                                    • Scans filesystem for orphaned metadata files
                                    • Repairs database paths automatically

🔍 SCANNING CONTROL:
  --force-rescan         Ignore existing database entries, scan everything fresh
                         • Recalculates all hashes and file information
                         • Use after file modifications or database corruption
                         
  --skip-folders         Skip directories that have been fully scanned (default: ON)
                         • Improves performance on large, stable directories
                         • Automatically skips folders with no new files
                         
  --no-skip-folders      Force scan ALL directories regardless of previous status
                         • Disables optimization, scans everything every time
                         • Use for thorough verification after major changes
                         
  --folder-limit N       Process only N folders per run (incremental processing)
                         • Perfect for huge directories (10,000+ models)
                         • Allows gradual processing without overwhelming system

🛠️  MAINTENANCE OPERATIONS:
  --migrate-paths OLD NEW      Update all database paths from OLD to NEW prefix
                               • Essential after moving model collections
                               • Example: "/old/path/" → "/new/path/"
                               
  --recover-missing           Automatically find files moved outside normal structure
                              • Searches alternative paths and destinations  
                              • Repairs database entries for relocated files
                              
  --retry-failed              Retry files that failed due to missing database columns
                              • Fixes issues after schema updates
                              • Reprocesses previously failed operations
                              
  --update-tables            Ensure all files have complete database entries
                             • Adds missing table entries for existing files
                             • Fixes incomplete scan results

🎚️  OUTPUT CONTROL (combines with ALL commands):
  --verbose              Show detailed operation information and progress
                         • File-by-file processing details
                         • Database operations and statistics
                         • Error diagnosis and resolution steps
                         
  --quiet                Minimize output to essential information only
                         • Opposite of --verbose
                         • Show only errors and final summaries
                         
  --dry-run              Preview ALL actions without making ANY changes
                         • 🚨 ESSENTIAL for --step sort operations
                         • Shows exactly what files would be moved where
                         • Generates detailed preview reports

RECOMMENDED COMMAND SEQUENCES:

New Collection Setup:
  1. **python model_sorter_main.py --step scan --verbose**
     → Discovers all model files and calculates SHA256/BLAKE3/AutoV3 hashes
     → Creates initial database entries with file paths and sizes
     → Foundation step required before all other operations
     
  2. **python model_sorter_main.py --extract-metadata-all --verbose**
     → Extracts model names, types, base models from SafeTensors headers and .civitai.info
     → Parses .metadata.json files and standardizes information across formats
     → Essential for intelligent organization and duplicate detection
     
  3. **python model_sorter_main.py --step duplicates --verbose**
     → Identifies identical models using hash comparison regardless of filename
     → Generates duplicate analysis report with metadata quality scoring
     → Shows which files to keep vs move based on metadata completeness
     
  4. **python model_sorter_main.py --step sort --dry-run --verbose**
     → Shows exactly what files would be moved where (preparation mode)
     → Previews directory structure and handles filename conflicts
     → Review output before proceeding to execution
     
  5. **python model_sorter_main.py --step sort --verbose**  # After reviewing preparation
     → Performs actual file organization into loras/base_model/model_name/ structure
     → Moves associated files (previews, metadata) alongside models
     → Creates final organized directory hierarchy

Daily Maintenance:
  • **python model_sorter_main.py --verbose**
    → Runs complete pipeline on any new files added since last run
    → Automatically detects and processes only unscanned files
    → Safe for daily execution as it won't re-process existing files
    
  • **python model_sorter_main.py --dry-run --verbose**
    → Scans and prepares processing for any new files without making changes
    → Shows what would be processed in a full run
    
  • **python model_sorter_main.py --rescan-linked --verbose**
    → Quick verification that model↔associated file relationships are intact
    → Fixes broken links caused by external file moves or renames
    → Recommended weekly to maintain database integrity
    
  • **python model_sorter_main.py --extract-metadata-limit 50**
    → Processes metadata for exactly 50 unprocessed models then stops
    → Perfect for gradual processing without overwhelming system resources
    → Use for large collections or when time is limited

Batch Processing (Large Collections):
  • **python model_sorter_main.py --step scan --folder-limit 10 --verbose**
    → Scans only 10 directories per run for gradual discovery
    → Prevents system overload with massive model collections (10,000+ files)
    → Run repeatedly until all directories are processed
    
  • **python model_sorter_main.py --extract-metadata-limit 100 --verbose**
    → Processes metadata for exactly 100 models per run
    → Allows controlled progress through large collections
    → Monitor progress and system resources between runs
    
  • **python model_sorter_main.py --step sort --dry-run --verbose**
    → Prepares organization results for current batch of processed models
    → Shows directory structure that would be created
    → Essential preparation step for large-scale operations
    
  • **python model_sorter_main.py --step sort --verbose**
    → Performs the organization after reviewing preparation
    → Commits the changes to disk

System Recovery:
  • **python model_sorter_main.py --recover-missing --verbose**
    → Searches for model files that were moved outside the normal structure
    → Checks alternative paths and destination directories for missing files
    → Updates database entries to reflect current file locations
    → Use when: Files show as "missing" but you know they still exist somewhere
    
  • **python model_sorter_main.py --retry-failed --verbose**
    → Retries metadata extraction for files that previously failed
    → Fixes issues caused by missing database columns or schema updates
    → Reprocesses files marked as "failed" in previous operations
    → Use when: After database schema updates or to fix incomplete processing
    
  • **python model_sorter_main.py --rescan-linked-rebuild --verbose**  # Last resort
    → Completely destroys and rebuilds ALL file associations from scratch
    → Scans entire filesystem to recreate model↔associated file relationships
    → Nuclear option that takes significant time but fixes corrupted associations
    → Use when: Database associations are completely broken or corrupted
    
  • **python model_sorter_main.py --migrate-paths "/old/path" "/new/path"**
    → Updates ALL database file paths from old prefix to new prefix
    → Essential after moving entire model collections to new locations
    → Batch updates thousands of database entries efficiently
    → Use when: You've moved your model directory structure to a new location

⚠️  CRITICAL SAFETY NOTES:
  🚨 ALWAYS use --dry-run with 'sort' step before live execution
  💾 Backup model_sorter.sqlite before major operations  
  📊 Use --verbose to understand system behavior
  📁 Reports saved to processing_report/ directory for review
  🔄 Test operations on small subsets before full runs

📁 Configuration File (config.ini):
  source_directory: Root directory to scan (e.g., /path/to/your/models/)
  destination_root: Organized file destination (configured in config.ini)
  civitai_database: Civitai reference database (Database/civitai.sqlite)
  max_file_size_mb: Maximum file size to process (default: 10240 MB = 10GB)
  database_commit_batch_size: Files per database transaction (default: 1000)

🎯 For detailed script documentation, see: script-core-functionality.md
        """)
    
    parser.add_argument('--config', '-c', default='config.ini',
                       help='Configuration file path (default: config.ini)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be done without executing file operations')
    parser.add_argument('--step', '-s', 
                       choices=['scan', 'metadata', 'duplicates', 'sort', 'civitai', 'rescan', 'extract-metadata', 'retry-failed', 'migrate-paths', 'recover-missing', 'force-blake3', 'update-tables', 'rescan-linked', 'rescan-linked-rebuild', 'rescan-and-repair-metadata-text'],
                       help='Run specific pipeline step: scan=discover files, metadata=extract info, duplicates=find identical files, sort=organize structure, civitai=generate info files. See detailed descriptions above.')
    parser.add_argument('--rescan-linked', action='store_true',
                       help='Quick incremental verification and repair of file associations')
    parser.add_argument('--rescan-linked-rebuild', action='store_true',
                       help='Complete rebuild of all file associations from scratch')
    parser.add_argument('--rescan-and-repair-metadata-text', action='store_true',
                       help='Intelligently scan filesystem and repair metadata/text file path mismatches')
    parser.add_argument('--force-rescan', '-f', action='store_true',
                       help='Force rescan of all files, ignoring database cache (use with --step scan)')
    parser.add_argument('--extract-metadata-limit', type=int, 
                       help='📊 BATCH MODE: Extract metadata from exactly N unprocessed files then stop. Perfect for gradual processing. Combines with: --verbose. Auto-runs extract-metadata step.')
    parser.add_argument('--extract-metadata-all', action='store_true',
                       help='📊 COMPLETE MODE: Extract metadata from ALL files missing metadata. Skips already processed files. Combines with: --verbose. Auto-runs extract-metadata step.')
    parser.add_argument('--extract-metadata-all-rescan', action='store_true',
                       help='📊 FORCE MODE: Re-extract metadata from ALL files including already processed ones. Use after metadata format changes. Combines with: --verbose. Auto-runs extract-metadata step.')
    parser.add_argument('--update-tables', action='store_true',
                       help='Check database completeness and rescan files with missing table entries')
    parser.add_argument('--skip-folders', action='store_true', default=True,
                       help='Skip directories that have already been fully scanned (default: enabled)')
    parser.add_argument('--no-skip-folders', action='store_false', dest='skip_folders',
                       help='Disable directory skipping (force scan all directories)')
    parser.add_argument('--folder-limit', type=int, metavar='N',
                       help='Limit scanning to N folders per run for incremental processing (useful for large directories)')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry metadata extraction for files that previously failed due to missing columns')
    parser.add_argument('--migrate-paths', nargs=2, metavar=('OLD_PREFIX', 'NEW_PREFIX'),
                       help='Migrate database paths from old prefix to new prefix (e.g., "/old/path/" "/new/path/")')
    parser.add_argument('--migrate-tables', action='store_true',
                       help='Migrate model files from scanned_files to model_files table while preserving associations')
    parser.add_argument('--recover-missing', action='store_true',
                       help='Automatically recover missing files by checking alternative paths and destinations')
    parser.add_argument('--force-blake3', action='store_true',
                       help='Force rescan of non-SafeTensors files to add BLAKE3 hashes to database')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='🔍 DETAILED OUTPUT: Show file-by-file operations, database stats, progress indicators, and diagnostic information. Combines with: ALL commands. Essential for troubleshooting.')
    parser.add_argument('--version', action='version', version='Model Sorter v1.0.0')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        print("Please create a config.ini file or specify the path with --config")
        sys.exit(1)
    
    # Auto-detect and update project path in config.ini
    auto_detect_project_path(args.config)
    
    # Initialize orchestrator
    orchestrator = ModelSorterOrchestrator(
        config_path=args.config,
        dry_run=args.dry_run,  # Use --dry-run flag directly
        verbose=args.verbose or not args.quiet,  # Enable verbose if --verbose is set OR if --quiet is NOT set
        force_rescan=args.force_rescan,
        extract_metadata_limit=args.extract_metadata_limit,
        extract_metadata_all=args.extract_metadata_all,
        extract_metadata_all_rescan=args.extract_metadata_all_rescan,
        retry_failed=args.retry_failed,
        skip_folders=args.skip_folders,
        folder_limit=args.folder_limit
    )
    
    try:
        success = False
        
        # Auto-detect step based on flags
        if not args.step and (args.extract_metadata_limit or args.extract_metadata_all or args.extract_metadata_all_rescan):
            # If extract-metadata flags are provided without explicit step, run extract-metadata
            args.step = 'extract-metadata'
            print("Auto-detected step: extract-metadata (based on --extract-metadata flags)")
        elif not args.step and args.retry_failed:
            # If retry-failed flag is provided without explicit step, run retry-failed
            args.step = 'retry-failed'
            print("Auto-detected step: retry-failed (based on --retry-failed flag)")
        elif not args.step and args.migrate_paths:
            # If migrate-paths flag is provided without explicit step, run migrate-paths
            args.step = 'migrate-paths'
            print("Auto-detected step: migrate-paths (based on --migrate-paths flag)")
        elif not args.step and args.recover_missing:
            # If recover-missing flag is provided without explicit step, run recover-missing
            args.step = 'recover-missing'
            print("Auto-detected step: recover-missing (based on --recover-missing flag)")
        elif not args.step and args.force_blake3:
            # If force-blake3 flag is provided without explicit step, run force-blake3
            args.step = 'force-blake3'
            print("Auto-detected step: force-blake3 (based on --force-blake3 flag)")
        elif not args.step and args.update_tables:
            # If update-tables flag is provided without explicit step, run update-tables
            args.step = 'update-tables'
            print("Auto-detected step: update-tables (based on --update-tables flag)")
        elif not args.step and args.rescan_linked:
            # If rescan-linked flag is provided without explicit step, run rescan-linked
            args.step = 'rescan-linked'
            print("Auto-detected step: rescan-linked (based on --rescan-linked flag)")
        elif not args.step and args.rescan_linked_rebuild:
            # If rescan-linked-rebuild flag is provided without explicit step, run rescan-linked-rebuild
            args.step = 'rescan-linked-rebuild'
            print("Auto-detected step: rescan-linked-rebuild (based on --rescan-linked-rebuild flag)")
        elif not args.step and args.rescan_and_repair_metadata_text:
            # If rescan-and-repair-metadata-text flag is provided without explicit step
            args.step = 'rescan-and-repair-metadata-text'
            print("Auto-detected step: rescan-and-repair-metadata-text (based on --rescan-and-repair-metadata-text flag)")
        elif not args.step and args.migrate_tables:
            # If migrate-tables flag is provided without explicit step
            args.step = 'migrate-tables'
            print("Auto-detected step: migrate-tables (based on --migrate-tables flag)")
        
        if args.step:
            # Special handling for steps that require arguments
            if args.step == 'migrate-paths':
                if not args.migrate_paths:
                    print("ERROR: --migrate-paths step requires --migrate-paths OLD_PREFIX NEW_PREFIX arguments")
                    sys.exit(1)
                old_prefix, new_prefix = args.migrate_paths
                success = orchestrator.step_migrate_paths(old_prefix, new_prefix)
            else:
                # Run individual step
                success = orchestrator.run_individual_step(args.step)
        else:
            # Run full workflow
            success = orchestrator.run_full_workflow()
        
        if success:
            print("\nProcessing completed successfully!")
            sys.exit(0)
        else:
            print("\nProcessing completed with errors!")
            sys.exit(1)
            
    except BrokenPipeError:
        # Handle broken pipe error (e.g., when output is piped to head)
        # Suppress the error and exit gracefully
        try:
            sys.stdout.close()
        except:
            pass
        try:
            sys.stderr.close() 
        except:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()