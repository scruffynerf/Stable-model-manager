#!/usr/bin/env python3
"""
File Scanner and Hasher
Core scanning component that generates SHA256 hashes for all files and detects model files
"""

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import configparser

# Try to import BLAKE3 for fast hashing of non-SafeTensors files
try:
    import blake3  # type: ignore
    BLAKE3_AVAILABLE = True
except ImportError:
    blake3 = None  # type: ignore
    BLAKE3_AVAILABLE = False
    print("⚠️  BLAKE3 not available. Install with: pip install blake3 for faster non-SafeTensors hashing")
import re

# Import image processing libraries
PIL_AVAILABLE = False
Image = None
TAGS = {}

try:
    from PIL import Image  # type: ignore
    from PIL.ExifTags import TAGS  # type: ignore
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Image metadata extraction will be limited.")

try:
    import struct
    STRUCT_AVAILABLE = True
except ImportError:
    STRUCT_AVAILABLE = False


def log_metadata_error(error_message: str, file_path: Optional[str] = None, context: Optional[str] = None):
    """Log metadata extraction errors to JSON file"""
    import datetime
    
    error_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "error": error_message,
        "file_path": file_path,
        "context": context
    }
    
    error_file = "metadata_extractor_errors.json"
    
    # Load existing errors or start fresh
    try:
        if os.path.exists(error_file):
            with open(error_file, 'r', encoding='utf-8') as f:
                errors = json.load(f)
        else:
            errors = []
    except Exception:
        errors = []
    
    # Add new error
    errors.append(error_entry)
    
    # Save back to file
    try:
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not write to error log: {e}")


def compute_autov3_hex(path: Path) -> Optional[str]:
    """
    Compute SHA-256 over the file bytes *after* the safetensors header (offset = header_size + 8).
    Returns full hex digest (not truncated); caller can take the first 12 chars for AutoV3.
    If file is not parseable or header missing, returns None.
    """
    try:
        with open(path, "rb") as f:
            header8 = f.read(8)
            if len(header8) < 8:
                return None
            header_size = int.from_bytes(header8, "little")
            offset = header_size + 8
            # If offset beyond file, return None
            f.seek(0, 2)
            filesize = f.tell()
            if offset >= filesize:
                return None
            # Compute sha256 starting at offset
            f.seek(offset)
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
            return h.hexdigest()
    except Exception:
        return None


def extract_image_metadata(file_path: str) -> Dict:
    """Extract comprehensive metadata from image files"""
    metadata = {
        'format': None,
        'width': None,
        'height': None,
        'mode': None,
        'has_exif': False,
        'exif_data': {},
        'ai_metadata': {},
        'file_size': 0,
        'creation_date': None,
        'software': None,
        'error': None
    }
    
    try:
        # Get file size
        metadata['file_size'] = os.path.getsize(file_path)
        
        if not PIL_AVAILABLE or Image is None:
            metadata['error'] = "PIL/Pillow not available"
            return metadata
        
        with Image.open(file_path) as img:
            # Basic image properties
            metadata['format'] = img.format
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['mode'] = img.mode
            
            # Extract EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                metadata['has_exif'] = True
                exif_dict = img._getexif()
                
                for tag_id, value in exif_dict.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Skip UserComment field
                    if tag == 'UserComment':
                        continue
                        
                    # Convert non-JSON serializable types
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                        # Handle IFDRational and similar fraction types
                        try:
                            value = float(value)
                        except:
                            value = str(value)
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        # Convert any other non-serializable types to string
                        value = str(value)
                    
                    metadata['exif_data'][tag] = value
                    
                    # Extract specific useful fields
                    if tag == 'DateTime':
                        metadata['creation_date'] = value
                    elif tag == 'Software':
                        metadata['software'] = value
            
            # Extract AI generation metadata from PNG tEXt chunks
            if img.format == 'PNG' and hasattr(img, 'text'):
                ai_fields = ['parameters', 'prompt', 'negative_prompt', 'steps', 'sampler', 
                           'cfg_scale', 'seed', 'model', 'workflow', 'comfyui']
                
                for key, value in img.text.items():
                    key_lower = key.lower()
                    if any(ai_field in key_lower for ai_field in ai_fields):
                        metadata['ai_metadata'][key] = value
                    
                    # Common AI metadata fields
                    if key_lower in ['parameters', 'prompt', 'negative prompt', 'steps', 'sampler', 
                                   'cfg scale', 'seed', 'size', 'model hash', 'model']:
                        metadata['ai_metadata'][key] = value
            
            # Extract WebP metadata
            elif img.format == 'WEBP' and hasattr(img, 'info'):
                for key, value in img.info.items():
                    if isinstance(key, str) and any(term in key.lower() for term in ['exif', 'xmp']):
                        metadata['exif_data'][key] = str(value)[:500]  # Limit length
    
    except Exception as e:
        metadata['error'] = str(e)
    
    return metadata


def extract_comprehensive_metadata(file_path: str, basic_metadata: Dict) -> Dict:
    """Extract comprehensive generation metadata from images/videos including component usage"""
    import re
    
    metadata = {
        # Basic Civitai info
        'civitai_id': None,
        'civitai_uuid': None,
        'prompt_text': None,
        'negative_prompt': None,
        'has_meta': 0,
        'has_positive_prompt': 0,
        
        # Generation parameters
        'steps': None,
        'sampler': None,
        'cfg_scale': None,
        'seed': None,
        'width': basic_metadata.get('width'),  # Copy from basic metadata
        'height': basic_metadata.get('height'),  # Copy from basic metadata
        'denoising_strength': None,
        'clip_skip': None,
        
        # Model information
        'model_name': None,
        'model_hash': None,
        'vae_name': None,
        'vae_hash': None,
        
        # Upscaling
        'hires_upscaler': None,
        'hires_steps': None,
        'hires_upscale': None,
        
        # Tool detection
        'generation_tool': None,
        
        # Component usage (will be stored separately)
        'components': []
    }
    
    try:
        # Extract Civitai ID and UUID from filename
        filename = os.path.basename(file_path)
        
        civitai_id_match = re.search(r'_(\d{6,})_', filename)
        if civitai_id_match:
            metadata['civitai_id'] = int(civitai_id_match.group(1))
        
        uuid_match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', filename, re.IGNORECASE)
        if uuid_match:
            metadata['civitai_uuid'] = uuid_match.group(1)
        
        # Extract generation parameters from AI metadata
        generation_text = ""
        if 'ai_metadata' in basic_metadata and basic_metadata['ai_metadata']:
            ai_meta = basic_metadata['ai_metadata']
            
            # Collect all text that might contain generation parameters
            for key, value in ai_meta.items():
                if isinstance(value, str):
                    generation_text += f"{key}: {value}\n"
        
        # Also check EXIF data
        if 'exif_data' in basic_metadata and basic_metadata['exif_data']:
            for key, value in basic_metadata['exif_data'].items():
                if isinstance(value, str) and any(term in value.lower() for term in ['steps:', 'sampler:', 'cfg scale:']):
                    generation_text += f"{key}: {value}\n"
        
        if generation_text:
            metadata.update(parse_generation_parameters(generation_text))
        
        # Set metadata flags
        if metadata['steps'] or metadata['sampler'] or metadata['prompt_text']:
            metadata['has_meta'] = 1
        else:
            metadata['has_meta'] = 2
        
        if metadata['prompt_text']:
            metadata['has_positive_prompt'] = 1
        else:
            metadata['has_positive_prompt'] = 2
    
    except Exception as e:
        print(f"Error extracting comprehensive metadata from {file_path}: {e}")
    
    return metadata


def parse_comfyui_workflow(text: str) -> Optional[Dict]:
    """Parse ComfyUI workflow JSON to extract generation parameters"""
    import json
    import re
    
    params = {}
    components = []
    
    try:
        # Extract JSON data from the text
        prompt_match = re.search(r'prompt:\s*(\{.*?\})\s*workflow:', text, re.DOTALL)
        if not prompt_match:
            return None
        
        prompt_json = json.loads(prompt_match.group(1))
        
        # Extract prompt text
        for node_id, node in prompt_json.items():
            if node.get('class_type') == 'CLIPTextEncode':
                inputs = node.get('inputs', {})
                if 'text' in inputs and inputs['text'].strip():
                    # Check if this is positive prompt (not negative)
                    title = node.get('_meta', {}).get('title', '')
                    if 'negative' not in title.lower():
                        params['prompt_text'] = inputs['text']
                    else:
                        params['negative_prompt'] = inputs['text']
            
            # Extract generation parameters
            elif node.get('class_type') == 'BasicScheduler':
                inputs = node.get('inputs', {})
                if 'steps' in inputs:
                    params['steps'] = inputs['steps']
            
            elif node.get('class_type') == 'KSamplerSelect':
                inputs = node.get('inputs', {})
                if 'sampler_name' in inputs:
                    params['sampler'] = inputs['sampler_name']
            
            elif node.get('class_type') == 'RandomNoise':
                inputs = node.get('inputs', {})
                if 'noise_seed' in inputs:
                    params['seed'] = inputs['noise_seed']
            
            elif node.get('class_type') == 'CFGGuider':
                inputs = node.get('inputs', {})
                if 'cfg' in inputs:
                    params['cfg_scale'] = inputs['cfg']
            
            elif node.get('class_type') == 'EmptyLatentImage':
                inputs = node.get('inputs', {})
                if 'width' in inputs and 'height' in inputs:
                    params['width'] = inputs['width']
                    params['height'] = inputs['height']
            
            elif node.get('class_type') == 'UNETLoader':
                inputs = node.get('inputs', {})
                if 'unet_name' in inputs:
                    params['model_name'] = inputs['unet_name']
            
            # Extract LoRA usage
            elif node.get('class_type') in ['LoraLoader', 'Lora Loader Stack (rgthree)']:
                inputs = node.get('inputs', {})
                
                if node.get('class_type') == 'LoraLoader':
                    if 'lora_name' in inputs and 'strength_model' in inputs:
                        components.append({
                            'type': 'lora',
                            'name': inputs['lora_name'],
                            'weight': inputs['strength_model']
                        })
                
                elif node.get('class_type') == 'Lora Loader Stack (rgthree)':
                    # Handle stacked LoRA loader
                    for i in range(1, 10):  # Check lora_01 through lora_09
                        lora_key = f'lora_{i:02d}'
                        strength_key = f'strength_{i:02d}'
                        if lora_key in inputs and strength_key in inputs:
                            if inputs[lora_key] and inputs[lora_key] != '':
                                components.append({
                                    'type': 'lora',
                                    'name': inputs[lora_key], 
                                    'weight': inputs[strength_key]
                                })
        
        params['generation_tool'] = 'ComfyUI'
        params['has_meta'] = 1 if params else 2
        params['has_positive_prompt'] = 1 if params.get('prompt_text') else 2
        params['components'] = components
        
        return params
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        print(f"Error parsing ComfyUI workflow: {e}")
        return None


def normalize_spaced_text(text: str) -> str:
    """Normalize text that has spaces between every character (some Automatic1111 exports)"""
    # Check if text appears to be spaced out (many single character words)
    words = text.split()
    single_char_words = [w for w in words if len(w) == 1 and (w.isalnum() or w in '.,:;!?')]
    
    # If more than 50% of words are single characters, it's likely spaced out
    if len(words) > 20 and len(single_char_words) / len(words) > 0.5:
        import re
        
        # Simple approach: remove spaces between single characters
        # Split into tokens and rejoin intelligently
        tokens = text.split()
        result = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if len(token) == 1 and token.isalnum():
                # Start collecting single characters
                word = token
                i += 1
                
                # Collect more single alphanumeric characters
                while i < len(tokens) and len(tokens[i]) == 1 and tokens[i].isalnum():
                    word += tokens[i]
                    i += 1
                
                result.append(word)
            else:
                result.append(token)
                i += 1
        
        # Join with spaces and clean up
        normalized = ' '.join(result)
        
        # Clean up common spacing issues
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = re.sub(r'\s+([,.:;!?()])', r'\1', normalized)  # Space before punctuation
        normalized = re.sub(r'([,.:;!?()])\s+', r'\1 ', normalized)  # Single space after punctuation
        
        # Fix parentheses spacing - no spaces inside parentheses
        normalized = re.sub(r'\(\s+', '(', normalized)  # Remove space after opening parenthesis
        normalized = re.sub(r'\s+\)', ')', normalized)  # Remove space before closing parenthesis
        
        # Fix common parameter separators
        normalized = re.sub(r':\s*([0-9.])', r':\1', normalized)  # No space between colon and numbers
        normalized = re.sub(r'(\d)\s+\.', r'\1.', normalized)  # Fix "1 . 5" to "1.5"
        normalized = re.sub(r'\.\s+(\d)', r'.\1', normalized)  # Fix ". 5" to ".5"
        
        # Fix joined words before known parameter names
        parameter_names = ['Steps', 'Sampler', 'CFGscale', 'Seed', 'Size', 'Model', 'Negativeprompt']
        for param in parameter_names:
            # Add space before parameter names that got joined with previous words
            pattern = r'([a-z])(' + param + r'):'
            normalized = re.sub(pattern, r'\1 \2:', normalized)
        
        return normalized.strip()
    
    return text


def parse_generation_parameters(text: str) -> Dict:
    """Parse generation parameters from text (Automatic1111, ComfyUI, etc.)"""
    import re
    import json
    
    # Normalize spaced-out text first
    text = normalize_spaced_text(text)
    
    params = {}
    components = []
    
    try:
        # Check if this is ComfyUI JSON workflow data
        if 'prompt:' in text and '{' in text:
            comfyui_data = parse_comfyui_workflow(text)
            if comfyui_data:
                return comfyui_data
        # Extract prompts (everything before "Negative prompt:" is positive prompt)
        prompt_match = re.search(r'^(.*?)(?:\.?Negative prompt:|Steps:)', text, re.DOTALL)
        if prompt_match:
            prompt_text = prompt_match.group(1).strip()
            # Remove parameter prefixes
            prompt_text = re.sub(r'^(Parameters|User Comment):\s*', '', prompt_text, flags=re.IGNORECASE)
            params['prompt_text'] = prompt_text
        
        # Extract negative prompt - handle case where it spans multiple lines
        neg_prompt_match = re.search(r'Negative prompt:\s*(.*?)(?=\n\s*[A-Z][a-z]+:|Steps:|$)', text, re.DOTALL)
        if neg_prompt_match:
            neg_prompt = neg_prompt_match.group(1).strip()
            # Clean up any trailing punctuation before parameters
            neg_prompt = re.sub(r'\s*[.\n]*\s*$', '', neg_prompt)
            params['negative_prompt'] = neg_prompt
        
        # Extract generation parameters (comprehensive list)
        param_patterns = {
            'steps': r'Steps:\s*(\d+)',
            'sampler': r'Sampler:\s*([^,\n]+)',
            'cfg_scale': r'CFG scale:\s*([\d.]+)',
            'seed': r'Seed:\s*(\d+)',
            'model_name': r'Model:\s*([^,\n]+)',
            'model_hash': r'Model hash:\s*([a-fA-F0-9]+)',
            'vae_name': r'VAE:\s*([^,\n]+)',
            'vae_hash': r'VAE hash:\s*([a-fA-F0-9]+)',
            'denoising_strength': r'Denoising strength:\s*([\d.]+)',
            'clip_skip': r'Clip skip:\s*(\d+)',
            'hires_upscaler': r'Hires upscaler:\s*([^,\n]+)',
            'hires_steps': r'Hires steps:\s*(\d+)',
            'hires_upscale': r'Hires upscale:\s*([\d.]+)',
            'batch_size': r'Batch size:\s*(\d+)',
            'batch_pos': r'Batch pos:\s*(\d+)',
            'eta': r'Eta:\s*([\d.]+)',
            'ensd': r'ENSD:\s*(\d+)',
            'face_restoration': r'Face restoration:\s*([^,\n]+)',
            'restore_faces': r'Restore faces:\s*(True|False)',
            'tiled_diffusion': r'Tiled Diffusion:\s*([^,\n]+)',
            'hires_resize_mode': r'Hires resize mode:\s*([^,\n]+)',
            'first_pass_size': r'First pass size:\s*(\d+x\d+)',
            'addnet_enabled': r'AddNet Enabled:\s*(True|False)',
            'addnet_module_1': r'AddNet Module 1:\s*([^,\n]+)',
            'addnet_model_1': r'AddNet Model 1:\s*([^,\n]+)',
            'addnet_weight_1': r'AddNet Weight 1:\s*([\d.]+)',
            'addnet_weight_a_1': r'AddNet Weight A 1:\s*([\d.]+)',
            'addnet_weight_b_1': r'AddNet Weight B 1:\s*([\d.]+)',
            'mask_blur': r'Mask blur:\s*([\d.]+)',
            'noise_multiplier': r'Noise multiplier:\s*([\d.]+)',
        }
        
        for param_name, pattern in param_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Integer parameters
                if param_name in ['steps', 'seed', 'clip_skip', 'hires_steps', 'batch_size', 'batch_pos', 'ensd']:
                    params[param_name] = int(value)
                # Float parameters - with safe parsing for malformed values
                elif param_name in ['cfg_scale', 'denoising_strength', 'hires_upscale', 'eta', 'addnet_weight_1', 
                                   'addnet_weight_a_1', 'addnet_weight_b_1', 'mask_blur', 'noise_multiplier']:
                    try:
                        # Try direct conversion first
                        params[param_name] = float(value)
                    except ValueError:
                        # Handle malformed values like "0.9:OUTD" - extract just the number part
                        import re
                        number_match = re.search(r'(\d+(?:\.\d+)?)', value)
                        if number_match:
                            try:
                                params[param_name] = float(number_match.group(1))
                            except ValueError:
                                log_metadata_error(f"Could not parse float value: '{value}' for parameter '{param_name}'", None, text[:100])
                        else:
                            log_metadata_error(f"Could not parse float value: '{value}' for parameter '{param_name}'", None, text[:100])
                # Boolean parameters
                elif param_name in ['restore_faces', 'addnet_enabled']:
                    params[param_name] = 1 if value.lower() == 'true' else 0
                # String parameters
                else:
                    params[param_name] = value
        
        # General pattern to catch any remaining metadata fields we might have missed
        # This looks for "Key: value" patterns that we haven't already captured
        # More restrictive to avoid matching prompt text
        general_pattern = r'\b([A-Z][A-Za-z0-9\s]*[A-Za-z0-9]):\s*([^,\n]+?)(?=,|\n[A-Z]|\n\s*$|$)'
        
        # Only apply general pattern to generation parameters section (after Steps:)
        # Split at "Steps:" to separate prompt text from generation parameters
        steps_split = re.split(r'\bSteps:', text, 1)
        if len(steps_split) > 1:
            # Only parse the generation parameters section, not the prompt text
            generation_params_text = 'Steps:' + steps_split[1]
            general_matches = re.findall(general_pattern, generation_params_text)
        else:
            # If no "Steps:" found, be very conservative - only match known parameter names
            known_params = r'\b(Model|VAE|Size|Sampler|CFG scale|Seed|Clip skip|Denoising strength|Hires upscaler|Hires steps|Hires upscale|ENSD|Batch size|Batch pos|Eta|AddNet|Workflow|Date):\s*([^,\n]+?)(?=,|\n|$)'
            general_matches = re.findall(known_params, text)
        
        for key, value in general_matches:
            # Clean up the key name to be database-friendly
            key_clean = key.strip().lower().replace(' ', '_').replace('-', '_')
            
            # Skip if we already have this parameter or it's problematic
            if (key_clean in params or 
                key_clean in ['parameters', 'prompt', 'negative_prompt'] or
                len(key_clean) < 2 or
                key_clean.startswith('negative_prompt') or
                # Skip common prompt words that might be mistaken for parameters
                key_clean in ['bad_anatomy', 'low_quality', 'worst_quality', 'normal_quality', 
                             'high_quality', 'very_detailed_background', 'masterpiece', 
                             'artist', 'lora', 'lyco', 'model', 'size'] or
                # Skip anything that contains newlines (corrupted parsing)
                '\n' in key or '\n' in value):
                continue
            
            # Additional validation: key should be a reasonable parameter name
            # (alphanumeric with underscores, no special chars)
            if not re.match(r'^[a-z][a-z0-9_]*$', key_clean):
                continue
            
            # Try to determine value type and convert
            value = value.strip()
            if re.match(r'^\d+$', value):
                params[key_clean] = int(value)
            elif re.match(r'^\d*\.\d+$', value):
                params[key_clean] = float(value)
            elif value.lower() in ['true', 'false']:
                params[key_clean] = 1 if value.lower() == 'true' else 0
            else:
                params[key_clean] = value
        
        # Extract size
        size_match = re.search(r'Size:\s*(\d+)x(\d+)', text)
        if size_match:
            params['width'] = int(size_match.group(1))
            params['height'] = int(size_match.group(2))
        
        # Extract component usage (LoRA, LyCO, etc.)
        lora_pattern = r'<lora:([^:>]+):([^>]+)>'
        lora_matches = re.findall(lora_pattern, text, re.IGNORECASE)
        for name, weight in lora_matches:
            # Safe weight parsing for malformed values like "0.7:MIDD"
            weight_value = 1.0
            if weight.strip():
                try:
                    weight_value = float(weight.strip())
                except ValueError:
                    # Extract just the number part
                    number_match = re.search(r'(\d+(?:\.\d+)?)', weight.strip())
                    if number_match:
                        try:
                            weight_value = float(number_match.group(1))
                        except ValueError:
                            log_metadata_error(f"Could not parse LoRA weight: '{weight.strip()}' for '{name.strip()}'", None, text[:100])
                            weight_value = 1.0
                    else:
                        log_metadata_error(f"Could not parse LoRA weight: '{weight.strip()}' for '{name.strip()}'", None, text[:100])
                        weight_value = 1.0
            
            components.append({
                'type': 'lora',
                'name': name.strip(),
                'weight': weight_value
            })
        
        lyco_pattern = r'<lyco:([^:>]+):([^>]+)>'
        lyco_matches = re.findall(lyco_pattern, text, re.IGNORECASE)
        for name, weight in lyco_matches:
            # Safe weight parsing for malformed values like "0.7:MIDD"
            weight_value = 1.0
            if weight.strip():
                try:
                    weight_value = float(weight.strip())
                except ValueError:
                    # Extract just the number part
                    number_match = re.search(r'(\d+(?:\.\d+)?)', weight.strip())
                    if number_match:
                        try:
                            weight_value = float(number_match.group(1))
                        except ValueError:
                            log_metadata_error(f"Could not parse LyCO weight: '{weight.strip()}' for '{name.strip()}'", None, text[:100])
                            weight_value = 1.0
                    else:
                        log_metadata_error(f"Could not parse LyCO weight: '{weight.strip()}' for '{name.strip()}'", None, text[:100])
                        weight_value = 1.0
            
            components.append({
                'type': 'lyco',
                'name': name.strip(),
                'weight': weight_value
            })
        
        # Detect generation tool
        if 'euler' in text.lower() or 'dpm++' in text.lower():
            params['generation_tool'] = 'Automatic1111'
        elif 'comfyui' in text.lower():
            params['generation_tool'] = 'ComfyUI'
        
        params['components'] = components
    
    except Exception as e:
        print(f"Error parsing generation parameters: {e}")
        log_metadata_error(f"Error parsing generation parameters: {e}", None, text[:200])
    
    return params


# Maintain backward compatibility
def extract_civitai_metadata(file_path: str, basic_metadata: Dict) -> Dict:
    """Legacy function - use extract_comprehensive_metadata instead"""
    return extract_comprehensive_metadata(file_path, basic_metadata)


class DatabaseManager:
    """Manages the scanner database for tracking scanned files"""
    
    def __init__(self, db_path: str = "model_sorter.sqlite"):
        self.db_path = db_path
        self.conn: sqlite3.Connection
        self._init_database()
    
    def _init_database(self):
        """Initialize the scanner database with required tables"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Scanned files table - tracks all scanned files
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scanned_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                autov3 TEXT,  -- AutoV3 hash for SafeTensors (model weights only)
                blake3 TEXT,  -- BLAKE3 hash for non-SafeTensors files (fast alternative)
                file_type TEXT NOT NULL,  -- 'model', 'image', 'text', 'other'
                extension TEXT NOT NULL,
                last_modified REAL NOT NULL,
                scan_date INTEGER NOT NULL,
                is_processed BOOLEAN DEFAULT 0,
                image_metadata TEXT,  -- JSON metadata for image files
                has_image_metadata BOOLEAN DEFAULT 0,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        # Metadata scan status table - tracks which metadata fields have been extracted
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata_scan_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_file_id INTEGER NOT NULL,
                field_name TEXT NOT NULL,
                scan_status INTEGER DEFAULT 0,  -- 0=not scanned, 1=scanned successfully, 2=scan failed, 3=field not applicable
                scan_date INTEGER DEFAULT (strftime('%s', 'now')),
                scan_notes TEXT,  -- Optional notes about the scan result
                field_value TEXT,  -- Stores the actual extracted value for quick lookup
                FOREIGN KEY (scanned_file_id) REFERENCES scanned_files (id),
                UNIQUE(scanned_file_id, field_name)
            )
        ''')
        
        # Model files table - specific tracking for model files
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_file_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                base_model TEXT,
                model_type TEXT,  -- 'LORA', 'Checkpoint', 'Embedding', etc.
                civitai_id INTEGER,
                version_id INTEGER,
                source_path TEXT NOT NULL,
                target_path TEXT,
                is_duplicate BOOLEAN DEFAULT 0,
                duplicate_group_id INTEGER,
                metadata_json TEXT,  -- JSON string of extracted metadata
                has_civitai_info BOOLEAN DEFAULT 0,
                has_metadata_json BOOLEAN DEFAULT 0,
                status TEXT DEFAULT 'pending',  -- 'pending', 'moved', 'duplicate_moved', 'error'
                error_message TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (scanned_file_id) REFERENCES scanned_files (id)
            )
        ''')
        
        # Associated files table - tracks files that go with models (images, metadata, etc)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS associated_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_file_id INTEGER NOT NULL,
                scanned_file_id INTEGER NOT NULL,
                association_type TEXT NOT NULL,  -- 'image', 'civitai_info', 'metadata', 'other'
                source_path TEXT NOT NULL,
                target_path TEXT,
                is_moved BOOLEAN DEFAULT 0,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (model_file_id) REFERENCES model_files (id),
                FOREIGN KEY (scanned_file_id) REFERENCES scanned_files (id)
            )
        ''')
        
        # Duplicate groups table - tracks groups of duplicate models
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicate_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sha256 TEXT NOT NULL,
                primary_model_id INTEGER,  -- The "best" copy (with most metadata/images)
                duplicate_count INTEGER DEFAULT 1,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (primary_model_id) REFERENCES model_files (id)
            )
        ''')
        
        # Processing log table - tracks processing operations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,  -- 'scan', 'move', 'duplicate_check', etc.
                file_path TEXT,
                status TEXT NOT NULL,  -- 'success', 'error', 'skipped'
                message TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        # Media metadata table - tracks Civitai-specific metadata for images and videos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_file_id INTEGER NOT NULL,
                civitai_id INTEGER,  -- Civitai image/video ID
                civitai_uuid TEXT,   -- UUID extracted from Civitai URL
                blur_hash TEXT,      -- BlurHash for visual similarity
                nsfw_level INTEGER,  -- NSFW rating
                minor_flag INTEGER DEFAULT 0,  -- 0=not scanned, 1=has minor content, 2=no minor content
                poi_flag INTEGER DEFAULT 0,     -- 0=not scanned, 1=person detected, 2=no person
                has_meta INTEGER DEFAULT 0,     -- 0=not scanned, 1=has generation metadata, 2=no metadata
                has_positive_prompt INTEGER DEFAULT 0,  -- 0=not scanned, 1=has positive prompt, 2=no prompt
                remix_of_id INTEGER, -- ID of original if this is a remix
                generation_params TEXT,  -- JSON string of extracted generation parameters
                prompt_text TEXT,    -- Extracted prompt text
                negative_prompt TEXT, -- Extracted negative prompt
                -- Core generation parameters
                steps INTEGER,       -- Generation steps
                sampler TEXT,        -- Sampler name (Euler a, DPM++ 2M Karras, etc.)
                cfg_scale REAL,      -- CFG Scale value
                seed INTEGER,        -- Generation seed
                width INTEGER,       -- Image width
                height INTEGER,      -- Image height
                denoising_strength REAL,  -- Denoising strength for img2img
                clip_skip INTEGER,   -- Clip skip value
                -- Model information
                model_name TEXT,     -- Base model name
                model_hash TEXT,     -- Model hash
                vae_name TEXT,       -- VAE name if specified
                vae_hash TEXT,       -- VAE hash if specified
                -- Upscaling information
                hires_upscaler TEXT, -- Hires upscaler name
                hires_steps INTEGER, -- Hires steps
                hires_upscale REAL,  -- Hires upscale factor
                -- Tool detection
                generation_tool TEXT, -- Tool used (Automatic1111, ComfyUI, etc.)
                scan_status INTEGER DEFAULT 0,  -- 0=not scanned, 1=partially scanned, 2=fully scanned
                last_scan_date INTEGER,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (scanned_file_id) REFERENCES scanned_files (id),
                UNIQUE(scanned_file_id)  -- One metadata record per file
            )
        ''')
        
        # Component usage table - tracks LoRA, LyCO, ControlNet, etc. usage in images
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS component_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_metadata_id INTEGER NOT NULL,
                component_type TEXT NOT NULL,  -- 'lora', 'lyco', 'controlnet', 'embedding', 'hypernetwork'
                component_name TEXT NOT NULL,  -- Name of the component
                component_weight REAL,         -- Weight/strength value
                component_hash TEXT,           -- Hash if available
                component_version TEXT,        -- Version if specified
                usage_context TEXT,            -- Additional context (e.g., ControlNet model, preprocessor)
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (media_metadata_id) REFERENCES media_metadata (id) ON DELETE CASCADE
            )
        ''')
        
        # Model components table - tracks available models, LoRAs, VAEs, etc. for cross-referencing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_type TEXT NOT NULL,  -- 'model', 'lora', 'lyco', 'vae', 'controlnet', 'embedding'
                component_name TEXT NOT NULL,  -- Display name
                file_path TEXT,                -- Path to component file if local
                component_hash TEXT,           -- Hash for identification
                civitai_id INTEGER,           -- Civitai model ID if known
                version_name TEXT,            -- Version name
                base_model TEXT,              -- Base model (SD 1.5, SDXL, etc.)
                description TEXT,             -- Description
                tags TEXT,                    -- JSON array of tags
                download_url TEXT,            -- Download URL if available
                file_size INTEGER,            -- File size in bytes
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(component_type, component_name, component_hash)
            )
        ''')
        
        # Schema migrations - add new columns if they don't exist
        migrations = [
            ('autov3', 'ALTER TABLE scanned_files ADD COLUMN autov3 TEXT'),
            ('blake3', 'ALTER TABLE scanned_files ADD COLUMN blake3 TEXT'),
            ('image_metadata', 'ALTER TABLE scanned_files ADD COLUMN image_metadata TEXT'),
            ('has_image_metadata', 'ALTER TABLE scanned_files ADD COLUMN has_image_metadata BOOLEAN DEFAULT 0')
        ]
        
        for column_name, migration_sql in migrations:
            try:
                cursor.execute(migration_sql)
                self.conn.commit()
            except sqlite3.OperationalError:
                # Column already exists
                pass
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scanned_files_sha256 ON scanned_files(sha256)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scanned_files_autov3 ON scanned_files(autov3)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scanned_files_blake3 ON scanned_files(blake3)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scanned_files_path ON scanned_files(file_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scanned_files_name_size ON scanned_files(file_name, file_size)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_files_sha256 ON model_files(scanned_file_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_duplicate_groups_sha256 ON duplicate_groups(sha256)')
        
        # Media metadata indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_metadata_civitai_id ON media_metadata(civitai_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_metadata_civitai_uuid ON media_metadata(civitai_uuid)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_metadata_blur_hash ON media_metadata(blur_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_metadata_scan_status ON media_metadata(scan_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_metadata_model_hash ON media_metadata(model_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_metadata_generation_tool ON media_metadata(generation_tool)')
        
        # Component usage indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_component_usage_media_id ON component_usage(media_metadata_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_component_usage_type ON component_usage(component_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_component_usage_name ON component_usage(component_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_component_usage_hash ON component_usage(component_hash)')
        
        # Model components indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_components_type ON model_components(component_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_components_name ON model_components(component_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_components_hash ON model_components(component_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_components_civitai_id ON model_components(civitai_id)')
        
        # Directory scan status table - tracks which directories are fully scanned
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS directory_scan_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                directory_path TEXT UNIQUE NOT NULL,
                scan_complete BOOLEAN DEFAULT 0,
                last_scan_date INTEGER,
                folder_mtime INTEGER,
                total_files INTEGER DEFAULT 0,
                scanned_files INTEGER DEFAULT 0,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        # Database completeness table - tracks which files have complete table entries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS database_completeness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_file_id INTEGER NOT NULL,
                has_scanned_files_entry BOOLEAN DEFAULT 1,
                has_media_metadata_entry BOOLEAN DEFAULT 0,
                has_component_usage_entries BOOLEAN DEFAULT 0,
                has_model_file_entry BOOLEAN DEFAULT 0,
                completeness_score REAL DEFAULT 0.0,
                last_check_date INTEGER DEFAULT (strftime('%s', 'now')),
                needs_rescan BOOLEAN DEFAULT 0,
                FOREIGN KEY (scanned_file_id) REFERENCES scanned_files (id),
                UNIQUE(scanned_file_id)
            )
        ''')
        
        # Schema migration: Add folder_mtime column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE directory_scan_status ADD COLUMN folder_mtime INTEGER")
            self.conn.commit()
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        # Additional indexes for new tables
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_directory_scan_path ON directory_scan_status(directory_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_directory_scan_complete ON directory_scan_status(scan_complete)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_completeness_scanned_file_id ON database_completeness(scanned_file_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_completeness_needs_rescan ON database_completeness(needs_rescan)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_completeness_score ON database_completeness(completeness_score)')
        
        self.conn.commit()
    
    def _ensure_media_metadata_columns(self, field_names: list) -> None:
        """Ensure all required columns exist in media_metadata table"""
        cursor = self.conn.cursor()
        
        # Get existing columns
        cursor.execute('PRAGMA table_info(media_metadata)')
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # Define column types for different field patterns
        def get_column_type(field_name: str) -> str:
            field_lower = field_name.lower()
            
            # Integer fields
            if any(pattern in field_lower for pattern in [
                'id', 'flag', 'level', 'steps', 'seed', 'width', 'height', 'clip_skip',
                'batch_size', 'batch_pos', 'ensd', 'hires_steps'
            ]):
                return 'INTEGER'
            
            # Real/Float fields  
            if any(pattern in field_lower for pattern in [
                'scale', 'strength', 'upscale', 'weight', 'eta', 'multiplier'
            ]):
                return 'REAL'
            
            # Boolean fields (stored as INTEGER)
            if any(pattern in field_lower for pattern in [
                'enabled', 'restore_faces', 'tiled'
            ]):
                return 'INTEGER'
                
            # Default to TEXT for everything else
            return 'TEXT'
        
        # Validate and sanitize field names
        def is_valid_column_name(name: str) -> bool:
            """Check if a column name is valid for SQLite"""
            if not name or len(name) > 64:  # Reasonable length limit
                return False
            # Must start with letter or underscore, contain only alphanumeric and underscores
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                return False
            # Cannot contain newlines or other problematic characters
            if '\n' in name or '\r' in name or '\0' in name:
                return False
            # SQLite reserved words to avoid
            reserved_words = {'table', 'index', 'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter'}
            if name.lower() in reserved_words:
                return False
            return True
        
        # Add missing columns
        for field_name in field_names:
            if field_name not in existing_columns and field_name != 'scanned_file_id':
                # Validate field name before attempting to create column
                if not is_valid_column_name(field_name):
                    print(f"Warning: Skipping invalid column name: {repr(field_name)}")
                    continue
                    
                column_type = get_column_type(field_name)
                try:
                    # Quote field name to handle SQL reserved words
                    quoted_field_name = f'"{field_name}"'
                    cursor.execute(f'ALTER TABLE media_metadata ADD COLUMN {quoted_field_name} {column_type}')
                    print(f"Added column to media_metadata: {field_name} {column_type}")
                except Exception as e:
                    print(f"Warning: Could not add column {field_name}: {e}")
                    log_metadata_error(f"Could not add column {field_name}: {e}", None, f"ALTER TABLE media_metadata ADD COLUMN {field_name} {column_type}")
        
        self.conn.commit()
    
    def _is_valid_field_name(self, name: str) -> bool:
        """Check if a field name is valid for database operations"""
        if not name or len(name) > 64:  # Reasonable length limit
            return False
        # Must start with letter or underscore, contain only alphanumeric and underscores
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return False
        # Cannot contain newlines or other problematic characters
        if '\n' in name or '\r' in name or '\0' in name:
            return False
        # SQLite reserved words to avoid
        reserved_words = {'table', 'index', 'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter'}
        if name.lower() in reserved_words:
            return False
        return True

    def _log_metadata_error(self, scanned_file_id: int, error: Exception, fields: dict):
        """Log metadata extraction errors to a JSON file for debugging"""
        import time
        from datetime import datetime
        
        error_log_path = "metadata_extractor_errors.json"
        
        # Get file path for context
        cursor = self.conn.cursor()
        cursor.execute('SELECT file_path FROM scanned_files WHERE id = ?', (scanned_file_id,))
        result = cursor.fetchone()
        file_path = result[0] if result else f"file_id_{scanned_file_id}"
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "scanned_file_id": scanned_file_id,
            "file_path": file_path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "fields_attempted": list(fields.keys()),
            "field_types": {k: type(v).__name__ for k, v in fields.items()}
        }
        
        # Load existing errors or create new list
        try:
            with open(error_log_path, 'r') as f:
                errors = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            errors = []
        
        # Add new error
        errors.append(error_entry)
        
        # Keep only last 1000 errors to prevent file from growing too large
        if len(errors) > 1000:
            errors = errors[-1000:]
        
        # Save updated errors
        try:
            with open(error_log_path, 'w') as f:
                json.dump(errors, f, indent=2)
        except Exception as write_error:
            print(f"Warning: Could not write to error log: {write_error}")

    def create_media_metadata(self, scanned_file_id: int, **kwargs) -> Optional[int]:
        """Create a new media metadata record with dynamic column support"""
        cursor = self.conn.cursor()
        
        # Filter out None values and invalid field names
        provided_fields = {}
        for k, v in kwargs.items():
            if v is not None and k != 'scanned_file_id':
                if self._is_valid_field_name(k):
                    # Convert lists to JSON strings for database storage
                    if isinstance(v, list):
                        provided_fields[k] = json.dumps(v)
                    else:
                        provided_fields[k] = v
                else:
                    print(f"Warning: Skipping invalid field name: {repr(k)}")
        
        if not provided_fields:
            # If no fields provided, just create basic record
            cursor.execute('''
                INSERT OR REPLACE INTO media_metadata (scanned_file_id, last_scan_date)
                VALUES (?, ?)
            ''', (scanned_file_id, int(time.time())))
            return cursor.lastrowid
        
        # Ensure all required columns exist
        self._ensure_media_metadata_columns(list(provided_fields.keys()))
        
        # Build and execute insert - quote field names to handle SQL reserved words
        quoted_fields = ['scanned_file_id'] + [f'"{field}"' for field in provided_fields.keys()] + ['last_scan_date']
        fields = ', '.join(quoted_fields)
        placeholders = ', '.join(['?'] * (len(provided_fields) + 2))
        values = [scanned_file_id] + list(provided_fields.values()) + [int(time.time())]

        try:
            cursor.execute(f'''
                INSERT OR REPLACE INTO media_metadata ({fields})
                VALUES ({placeholders})
            ''', values)
        except Exception as e:
            # Log error to file
            self._log_metadata_error(scanned_file_id, e, provided_fields)
            print(f"Error inserting media metadata for file {scanned_file_id}: {e}")
            print(f"Fields: {list(provided_fields.keys())}")
            # Try to add missing columns and retry
            self._ensure_media_metadata_columns(list(provided_fields.keys()))
            cursor.execute(f'''
                INSERT OR REPLACE INTO media_metadata ({fields})
                VALUES ({placeholders})
            ''', values)
        
        return cursor.lastrowid
    
    def get_media_metadata(self, scanned_file_id: int) -> dict:
        """Get media metadata for a scanned file"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM media_metadata WHERE scanned_file_id = ?
        ''', (scanned_file_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(zip([col[0] for col in cursor.description], row))
        return {}
    
    def update_media_metadata(self, scanned_file_id: int, **kwargs):
        """Update media metadata for a scanned file with dynamic column support"""
        cursor = self.conn.cursor()
        
        # Filter out None values and invalid field names
        provided_fields = {}
        for k, v in kwargs.items():
            if v is not None:
                if self._is_valid_field_name(k):
                    # Convert lists to JSON strings for database storage
                    if isinstance(v, list):
                        provided_fields[k] = json.dumps(v)
                    else:
                        provided_fields[k] = v
                else:
                    print(f"Warning: Skipping invalid field name: {repr(k)}")
        
        if provided_fields:
            # Ensure columns exist before trying to update
            self._ensure_media_metadata_columns(list(provided_fields.keys()))
            
            # Quote field names to handle SQL reserved words
            set_clause = ', '.join([f'"{k}" = ?' for k in provided_fields.keys()])
            values = list(provided_fields.values()) + [int(time.time()), scanned_file_id]
            
            try:
                cursor.execute(f'''
                    UPDATE media_metadata 
                    SET {set_clause}, updated_at = ?
                    WHERE scanned_file_id = ?
                ''', values)
            except Exception as e:
                # Log error to file
                self._log_metadata_error(scanned_file_id, e, provided_fields)
                print(f"Error updating media metadata for file {scanned_file_id}: {e}")
                print(f"Fields: {list(provided_fields.keys())}")
                # Ensure columns exist and retry
                self._ensure_media_metadata_columns(list(provided_fields.keys()))
                cursor.execute(f'''
                    UPDATE media_metadata 
                    SET {set_clause}, updated_at = ?
                    WHERE scanned_file_id = ?
                ''', values)
    
    def get_files_needing_metadata_scan(self, limit: Optional[int] = 100, force_rescan: bool = False) -> list:
        """Get files that need metadata scanning (images/videos)"""
        cursor = self.conn.cursor()
        
        # Build WHERE clause and ORDER BY based on force_rescan
        if force_rescan:
            # Include all files, even already scanned ones (scan_status = 1)
            where_clause = "sf.file_type IN ('image', 'video')"
            # When rescanning, prioritize successfully scanned files (status=1) first, then by scan date
            order_clause = "CASE WHEN mm.scan_status = 1 THEN 0 ELSE 1 END, COALESCE(mm.last_scan_date, 0) ASC"
        else:
            # Only files not yet scanned successfully (scan_status = 0 or NULL) or failed/not found files
            # Skip scan_status = 1 (successfully scanned) unless force_rescan
            where_clause = "sf.file_type IN ('image', 'video') AND COALESCE(mm.scan_status, 0) != 1"
            # Normal scan: order by oldest scan date first
            order_clause = "COALESCE(mm.last_scan_date, 0) ASC"
        
        if limit is None:
            # No limit - get all files
            cursor.execute(f'''
                SELECT sf.id, sf.file_path, sf.file_name, sf.file_type,
                       COALESCE(mm.scan_status, 0) as scan_status,
                       COALESCE(mm.last_scan_date, 0) as last_scan_date
                FROM scanned_files sf
                LEFT JOIN media_metadata mm ON sf.id = mm.scanned_file_id
                WHERE {where_clause}
                ORDER BY {order_clause}
            ''')
        else:
            # Apply limit
            cursor.execute(f'''
                SELECT sf.id, sf.file_path, sf.file_name, sf.file_type,
                       COALESCE(mm.scan_status, 0) as scan_status,
                       COALESCE(mm.last_scan_date, 0) as last_scan_date
                FROM scanned_files sf
                LEFT JOIN media_metadata mm ON sf.id = mm.scanned_file_id
                WHERE {where_clause}
                ORDER BY {order_clause}
                LIMIT ?
            ''', (limit,))
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def add_component_usage(self, media_metadata_id: int, component_type: str, 
                           component_name: str, component_weight: Optional[float] = None,
                           component_hash: Optional[str] = None, component_version: Optional[str] = None,
                           usage_context: Optional[str] = None) -> Optional[int]:
        """Add a component usage record (LoRA, LyCO, ControlNet, etc.)"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO component_usage 
            (media_metadata_id, component_type, component_name, component_weight, 
             component_hash, component_version, usage_context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (media_metadata_id, component_type, component_name, component_weight,
              component_hash, component_version, usage_context))
        return cursor.lastrowid
    
    def get_component_usage_by_media(self, media_metadata_id: int) -> list:
        """Get all component usage for a media file"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM component_usage WHERE media_metadata_id = ?
            ORDER BY component_type, component_name
        ''', (media_metadata_id,))
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def add_model_component(self, component_type: str, component_name: str,
                           file_path: Optional[str] = None, component_hash: Optional[str] = None,
                           civitai_id: Optional[int] = None, version_name: Optional[str] = None,
                           base_model: Optional[str] = None, description: Optional[str] = None,
                           tags: Optional[str] = None, download_url: Optional[str] = None,
                           file_size: Optional[int] = None) -> Optional[int]:
        """Add or update a model component record"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO model_components 
            (component_type, component_name, file_path, component_hash, civitai_id,
             version_name, base_model, description, tags, download_url, file_size,
             updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
        ''', (component_type, component_name, file_path, component_hash, civitai_id,
              version_name, base_model, description, tags, download_url, file_size))
        return cursor.lastrowid
    
    def find_component_by_name(self, component_name: str, component_type: Optional[str] = None) -> list:
        """Find components by name, optionally filtered by type"""
        cursor = self.conn.cursor()
        if component_type:
            cursor.execute('''
                SELECT * FROM model_components 
                WHERE component_name LIKE ? AND component_type = ?
                ORDER BY component_name
            ''', (f'%{component_name}%', component_type))
        else:
            cursor.execute('''
                SELECT * FROM model_components 
                WHERE component_name LIKE ?
                ORDER BY component_type, component_name
            ''', (f'%{component_name}%',))
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_component_usage_stats(self) -> dict:
        """Get statistics on component usage"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                component_type,
                COUNT(*) as usage_count,
                COUNT(DISTINCT component_name) as unique_components
            FROM component_usage
            GROUP BY component_type
            ORDER BY usage_count DESC
        ''')
        
        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {
                'usage_count': row[1],
                'unique_components': row[2]
            }
        
        return stats
    
    def find_model_by_hash(self, model_hash: str) -> dict:
        """Find a model file by its hash in the scanned files database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT sf.*, mf.model_type, mf.base_model 
            FROM scanned_files sf
            LEFT JOIN model_files mf ON sf.id = mf.scanned_file_id
            WHERE sf.sha256 = ? AND sf.file_type = 'model'
        ''', (model_hash.upper(),))
        
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        return {}
    
    def find_component_by_hash(self, component_hash: str, component_type: Optional[str] = None) -> dict:
        """Find a component by hash in model_components table"""
        cursor = self.conn.cursor()
        if component_type:
            cursor.execute('''
                SELECT * FROM model_components 
                WHERE component_hash = ? AND component_type = ?
            ''', (component_hash, component_type))
        else:
            cursor.execute('''
                SELECT * FROM model_components 
                WHERE component_hash = ?
            ''', (component_hash,))
        
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        return {}
    
    def mark_media_ignored(self, scanned_file_id: int, reason: str = "No metadata or cross-reference found"):
        """Mark a media file to be ignored in future scans"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE media_metadata 
            SET scan_status = -1, updated_at = strftime('%s', 'now')
            WHERE scanned_file_id = ?
        ''', (scanned_file_id,))
        
        # Log the reason
        self.log_operation('ignore_media', '', 'info', f'File ID {scanned_file_id}: {reason}')
    
    def update_scanned_file_location(self, scanned_file_id: int, new_path: str):
        """Update the file path for a scanned file"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE scanned_files 
            SET file_path = ?, file_name = ?
            WHERE id = ?
        ''', (new_path, os.path.basename(new_path), scanned_file_id))
        self.conn.commit()
    
    def get_media_by_blurhash(self, blur_hash: str) -> list:
        """Find media files by BlurHash (for future civitai cross-reference)"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT sf.*, mm.*
            FROM scanned_files sf
            JOIN media_metadata mm ON sf.id = mm.scanned_file_id
            WHERE mm.blur_hash = ?
        ''', (blur_hash,))
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def add_scanned_file(self, file_path: str, file_name: str, file_size: int, 
                        sha256: str, file_type: str, extension: str, 
                        last_modified: float, autov3: Optional[str] = None, 
                        blake3_hash: Optional[str] = None,
                        image_metadata: Optional[Dict] = None) -> Optional[int]:
        """Add a scanned file record and return the ID"""
        cursor = self.conn.cursor()
        scan_date = int(time.time())
        
        # Convert image metadata to JSON string
        image_metadata_json = None
        has_image_metadata = False
        if image_metadata:
            image_metadata_json = json.dumps(image_metadata)
            has_image_metadata = True
        
        cursor.execute('''
            INSERT OR REPLACE INTO scanned_files 
            (file_path, file_name, file_size, sha256, autov3, blake3, file_type, extension, 
             last_modified, scan_date, image_metadata, has_image_metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (file_path, file_name, file_size, sha256, autov3, blake3_hash, file_type, extension,
              last_modified, scan_date, image_metadata_json, has_image_metadata, scan_date))
        
        # Don't commit here - let FileScanner handle batched commits
        return cursor.lastrowid
    
    def commit(self):
        """Manually commit the database transaction"""
        self.conn.commit()
    
    def file_exists_by_name_size(self, file_name: str, file_size: int) -> bool:
        """Check if file exists by name and size (fast skip check)"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT 1 FROM scanned_files WHERE file_name = ? AND file_size = ?', 
                      (file_name, file_size))
        return cursor.fetchone() is not None
    
    def get_file_by_hash(self, sha256: str) -> Optional[dict]:
        """Get file record by SHA256 hash"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM scanned_files WHERE sha256 = ?', (sha256,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_file_by_autov3(self, autov3: str) -> Optional[dict]:
        """Get file record by AutoV3 hash"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM scanned_files WHERE autov3 = ?', (autov3,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_files_by_autov3(self, autov3: str) -> List[dict]:
        """Get all file records with the same AutoV3 hash"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM scanned_files WHERE autov3 = ?', (autov3,))
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def get_file_by_blake3(self, blake3_hash: str) -> Optional[dict]:
        """Get file record by BLAKE3 hash"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM scanned_files WHERE blake3 = ?', (blake3_hash,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_files_by_blake3(self, blake3_hash: str) -> List[dict]:
        """Get all file records with the same BLAKE3 hash"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM scanned_files WHERE blake3 = ?', (blake3_hash,))
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def is_file_already_scanned(self, file_path: str, last_modified: float) -> bool:
        """Check if file has already been scanned and is up to date"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT last_modified FROM scanned_files 
            WHERE file_path = ? AND last_modified = ?
        ''', (file_path, last_modified))
        return cursor.fetchone() is not None
    
    def get_file_by_path(self, file_path: str) -> Optional[dict]:
        """Get file record by file path"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM scanned_files WHERE file_path = ?', (file_path,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_files_sample(self, limit: int = 100) -> List[dict]:
        """Get a sample of files from the database for checking"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM scanned_files LIMIT ?', (limit,))
        rows = cursor.fetchall()
        
        if rows:
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        return []
    
    def get_files_missing_metadata(self) -> List[Dict]:
        """Get files that are missing metadata based on their type"""
        cursor = self.conn.cursor()
        
        # Get image files without image metadata
        cursor.execute('''
            SELECT * FROM scanned_files 
            WHERE file_type = 'image' 
              AND (has_image_metadata = 0 OR image_metadata IS NULL)
              AND extension IN ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp')
        ''')
        
        missing_files = []
        columns = [desc[0] for desc in cursor.description]
        
        for row in cursor.fetchall():
            file_dict = dict(zip(columns, row))
            file_dict['missing_metadata_type'] = 'image'
            missing_files.append(file_dict)
        
        # Get SafeTensors files without AutoV3 hash
        cursor.execute('''
            SELECT * FROM scanned_files 
            WHERE file_type = 'model' 
              AND extension = '.safetensors'
              AND (autov3 IS NULL OR autov3 = '')
        ''')
        
        for row in cursor.fetchall():
            file_dict = dict(zip(columns, row))
            file_dict['missing_metadata_type'] = 'autov3'
            missing_files.append(file_dict)
        
        return missing_files
    
    def update_file_metadata(self, file_id: int, autov3: Optional[str] = None, 
                           image_metadata: Optional[Dict] = None):
        """Update metadata for an existing file"""
        cursor = self.conn.cursor()
        update_fields = []
        values = []
        
        if autov3 is not None:
            update_fields.append('autov3 = ?')
            values.append(autov3)
        
        if image_metadata is not None:
            update_fields.append('image_metadata = ?')
            update_fields.append('has_image_metadata = ?')
            values.append(json.dumps(image_metadata))
            values.append(True)
        
        if update_fields:
            update_fields.append('updated_at = ?')
            values.append(int(time.time()))
            values.append(file_id)
            
            sql = f"UPDATE scanned_files SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(sql, values)
            self.conn.commit()
    
    def update_blake3_hash(self, file_id: int, blake3_hash: str):
        """Update BLAKE3 hash for an existing file"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE scanned_files 
            SET blake3 = ?, updated_at = ? 
            WHERE id = ?
        ''', (blake3_hash, int(time.time()), file_id))
        # Note: Don't commit here for batch processing efficiency
    
    def get_files_needing_blake3(self, directory_path: Optional[str] = None) -> List[Dict]:
        """Get files that need BLAKE3 hashes (non-SafeTensors files without BLAKE3)"""
        cursor = self.conn.cursor()
        
        if directory_path:
            # Only get files within the specified directory
            cursor.execute('''
                SELECT id, file_path, file_name, file_size 
                FROM scanned_files 
                WHERE (blake3 IS NULL OR blake3 = '') 
                  AND file_path NOT LIKE '%.safetensors'
                  AND file_path LIKE ?
                ORDER BY file_path
            ''', (f"{directory_path}%",))
        else:
            # Get all files needing BLAKE3
            cursor.execute('''
                SELECT id, file_path, file_name, file_size 
                FROM scanned_files 
                WHERE (blake3 IS NULL OR blake3 = '') 
                  AND file_path NOT LIKE '%.safetensors'
                ORDER BY file_path
            ''')
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def update_directory_scan_status(self, directory_path: str, scan_complete: bool = True, 
                                   total_files: int = 0, scanned_files: int = 0):
        """Update directory scan completion status"""
        import os
        cursor = self.conn.cursor()
        
        # Get folder modification time
        folder_mtime = None
        try:
            if os.path.exists(directory_path):
                folder_mtime = int(os.path.getmtime(directory_path))
        except (OSError, ValueError):
            folder_mtime = None
        
        cursor.execute('''
            INSERT OR REPLACE INTO directory_scan_status 
            (directory_path, scan_complete, last_scan_date, folder_mtime, total_files, scanned_files, updated_at)
            VALUES (?, ?, strftime('%s', 'now'), ?, ?, ?, strftime('%s', 'now'))
        ''', (directory_path, scan_complete, folder_mtime, total_files, scanned_files))
        # Note: Don't commit here for batch processing efficiency
    
    def is_directory_fully_scanned(self, directory_path: str) -> bool:
        """Check if directory is marked as fully scanned and hasn't been modified since last scan"""
        import os
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT scan_complete, folder_mtime, last_scan_date FROM directory_scan_status 
            WHERE directory_path = ? AND scan_complete = 1
        ''', (directory_path,))
        
        result = cursor.fetchone()
        if not result:
            return False
        
        scan_complete, stored_mtime, last_scan_date = result
        
        # Check if folder has been modified since last scan
        try:
            if os.path.exists(directory_path):
                current_mtime = int(os.path.getmtime(directory_path))
                
                # If we have stored mtime, compare with current
                if stored_mtime is not None:
                    if current_mtime > stored_mtime:
                        # Folder has been modified since last scan
                        return False
                
                # Additional check: if current mtime is newer than last scan date
                if last_scan_date and current_mtime > last_scan_date:
                    return False
        except (OSError, ValueError, TypeError):
            # If we can't get mtime, assume it needs rescanning
            return False
        
        return scan_complete == 1

    def check_directory_for_changes(self, directory_path: str) -> tuple[bool, str]:
        """Check if directory has changes and return status with reason"""
        import os
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT scan_complete, folder_mtime, last_scan_date FROM directory_scan_status 
            WHERE directory_path = ?
        ''', (directory_path,))
        
        result = cursor.fetchone()
        if not result:
            return False, "new_directory"
        
        scan_complete, stored_mtime, last_scan_date = result
        
        if not scan_complete:
            return False, "incomplete"
        
        # Check if folder has been modified since last scan
        try:
            if os.path.exists(directory_path):
                current_mtime = int(os.path.getmtime(directory_path))
                
                # If we have stored mtime, compare with current
                if stored_mtime is not None:
                    if current_mtime > stored_mtime:
                        return False, "additional_files_found"
                
                # Additional check: if current mtime is newer than last scan date
                if last_scan_date and current_mtime > last_scan_date:
                    return False, "additional_files_found"
        except (OSError, ValueError, TypeError):
            return False, "mtime_error"
        
        return True, "up_to_date"
    
    def get_incomplete_directories(self) -> List[Dict]:
        """Get directories that need scanning or rescanning"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT directory_path, scan_complete, total_files, scanned_files, last_scan_date
            FROM directory_scan_status 
            WHERE scan_complete = 0 OR total_files != scanned_files
            ORDER BY directory_path
        ''')
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_directory_scan_info(self, directory_path: str) -> Dict:
        """Get scan information for a specific directory"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT directory_path, scan_complete, total_files, scanned_files, last_scan_date
            FROM directory_scan_status 
            WHERE directory_path = ?
        ''', (directory_path,))
        
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        return {}
    
    def check_database_completeness(self, scanned_file_id: int) -> Dict:
        """Check completeness of database entries for a file"""
        cursor = self.conn.cursor()
        
        # Get file info
        cursor.execute('SELECT file_type, extension FROM scanned_files WHERE id = ?', (scanned_file_id,))
        file_info = cursor.fetchone()
        if not file_info:
            return {'complete': False, 'score': 0.0, 'missing': ['file_not_found']}
        
        file_type, extension = file_info
        missing_entries = []
        score = 0.0
        
        # Check media_metadata entry (for images)
        if file_type == 'image' and extension.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
            cursor.execute('SELECT id FROM media_metadata WHERE scanned_file_id = ?', (scanned_file_id,))
            if cursor.fetchone():
                score += 0.4
            else:
                missing_entries.append('media_metadata')
            
            # Check component_usage entries
            cursor.execute('''
                SELECT COUNT(*) FROM component_usage cu 
                JOIN media_metadata mm ON cu.media_metadata_id = mm.id 
                WHERE mm.scanned_file_id = ?
            ''', (scanned_file_id,))
            component_count = cursor.fetchone()[0]
            if component_count > 0:
                score += 0.3
            else:
                missing_entries.append('component_usage')
        
        # Check model_files entry (for models)
        elif file_type == 'model':
            cursor.execute('SELECT id FROM model_files WHERE scanned_file_id = ?', (scanned_file_id,))
            if cursor.fetchone():
                score += 0.5
            else:
                missing_entries.append('model_files')
        
        # Base score for having scanned_files entry
        score += 0.3
        
        return {
            'complete': len(missing_entries) == 0,
            'score': score,
            'missing': missing_entries
        }
    
    def update_completeness_tracking(self, scanned_file_id: int):
        """Update completeness tracking for a file"""
        completeness = self.check_database_completeness(scanned_file_id)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO database_completeness 
            (scanned_file_id, has_media_metadata_entry, has_component_usage_entries, 
             has_model_file_entry, completeness_score, needs_rescan, last_check_date)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
        ''', (
            scanned_file_id,
            'media_metadata' not in completeness['missing'],
            'component_usage' not in completeness['missing'],
            'model_files' not in completeness['missing'],
            completeness['score'],
            not completeness['complete']
        ))
        self.conn.commit()
    
    def get_files_needing_rescan(self) -> List[Dict]:
        """Get files that need rescanning due to incomplete database entries"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT sf.id, sf.file_path, sf.file_name, sf.file_type, 
                   dc.completeness_score, dc.last_check_date
            FROM scanned_files sf
            JOIN database_completeness dc ON sf.id = dc.scanned_file_id
            WHERE dc.needs_rescan = 1
            ORDER BY dc.completeness_score ASC, sf.file_path
        ''')
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def log_operation(self, operation_type: str, file_path: str, 
                     status: str, message: str = ""):
        """Log a processing operation"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO processing_log (operation_type, file_path, status, message)
            VALUES (?, ?, ?, ?)
        ''', (operation_type, file_path, status, message))
        self.conn.commit()


def read_metadata_text_file(file_path: str) -> Dict:
    """Read and parse metadata from a text file"""
    import re
    
    metadata = {
        'found_components': [],
        'model_references': [],
        'raw_text': ''
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            metadata['raw_text'] = content
        
        # Parse using the same generation parameter parser
        parsed = parse_generation_parameters(content)
        
        # Extract component references
        if 'components' in parsed:
            metadata['found_components'] = parsed['components']
        
        # Look for model hash references
        model_hash_pattern = r'Model hash:\s*([a-fA-F0-9]+)'
        model_hashes = re.findall(model_hash_pattern, content, re.IGNORECASE)
        
        # Look for model name references
        model_name_pattern = r'Model:\s*([^,\n]+)'
        model_names = re.findall(model_name_pattern, content, re.IGNORECASE)
        
        for hash_val in model_hashes:
            metadata['model_references'].append({'type': 'hash', 'value': hash_val.strip()})
        
        for name in model_names:
            metadata['model_references'].append({'type': 'name', 'value': name.strip()})
        
        return metadata
    
    except Exception as e:
        print(f"Error reading metadata file {file_path}: {e}")
        return metadata


def find_metadata_text_file(media_file_path: str) -> str:
    """Find associated metadata text file for a media file"""
    import glob
    
    base_path = os.path.splitext(media_file_path)[0]
    directory = os.path.dirname(media_file_path)
    filename = os.path.basename(base_path)
    
    # Common metadata file patterns
    patterns = [
        f"{base_path}.txt",
        f"{base_path}.metadata.txt",
        f"{base_path}.info.txt",
        f"{directory}/{filename}_metadata.txt",
        f"{directory}/{filename}_info.txt",
        f"{directory}/metadata.txt",
        f"{directory}/info.txt"
    ]
    
    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern
    
    return ""


def cross_reference_with_civitai(sha256_hash: str, blur_hash: Optional[str] = None, civitai_db_path: str = "Database/civitai.sqlite") -> Dict:
    """Cross-reference image hashes with civitai database to find related models"""
    import sqlite3
    
    results = {
        'found_models': [],
        'found_components': [],
        'civitai_matches': []
    }
    
    try:
        # Connect to civitai database
        civitai_conn = sqlite3.connect(civitai_db_path)
        cursor = civitai_conn.cursor()
        
        # Search by SHA256 hash in model_files table (exact match)
        cursor.execute('''
            SELECT mf.id, mf.model_id, mf.version_id, mf.type, mf.sha256, mf.data,
                   m.name, m.type as model_type
            FROM model_files mf
            JOIN models m ON mf.model_id = m.id
            WHERE mf.sha256 = ?
            LIMIT 5
        ''', (sha256_hash,))
        
        for row in cursor.fetchall():
            try:
                import json
                data = json.loads(row[5]) if row[5] else {}
                results['civitai_matches'].append({
                    'file_id': row[0],
                    'model_id': row[1],
                    'version_id': row[2],
                    'file_type': row[3],
                    'sha256': row[4],
                    'name': row[6],
                    'model_type': row[7],  # This is the 'type' field from models table
                    'data': data,
                    'type': 'sha256_match'
                })
                if civitai_db_path and "/ComfyUI-Lora-Manager-main/" in civitai_db_path:
                    print(f"  Found in civitai database - correcting metadata")
            except Exception as e:
                print(f"Warning: Could not parse civitai data: {e}")
                pass

        # Search by BlurHash if available (in model data)
        if blur_hash:
            cursor.execute('''
                SELECT mf.id, mf.model_id, mf.version_id, mf.type, mf.sha256, mf.data,
                       m.name, m.type as model_type
                FROM model_files mf
                JOIN models m ON mf.model_id = m.id
                WHERE mf.data LIKE ?
                LIMIT 10
            ''', (f'%{blur_hash}%',))
            
            for row in cursor.fetchall():
                try:
                    import json
                    data = json.loads(row[5]) if row[5] else {}
                    results['civitai_matches'].append({
                        'file_id': row[0],
                        'model_id': row[1],
                        'version_id': row[2],
                        'file_type': row[3],
                        'sha256': row[4],
                        'name': row[6],
                        'model_type': row[7],
                        'data': data,
                        'type': 'blurhash_match'
                    })
                except Exception as e:
                    print(f"Warning: Could not parse civitai blurhash data: {e}")
                    pass
        
        civitai_conn.close()
        
    except Exception as e:
        print(f"Error cross-referencing with civitai database: {e}")
    
    return results


# Add incremental metadata tracking methods to DatabaseManager
def add_metadata_methods_to_database_manager():
    """Add metadata tracking methods to DatabaseManager class"""
    
    def get_metadata_scan_status(self, scanned_file_id: int, field_name: Optional[str] = None) -> Optional[Dict]:
        """Get metadata scan status for a file and optionally specific field"""
        cursor = self.conn.cursor()
        
        if field_name:
            cursor.execute('''
                SELECT field_name, scan_status, scan_date, scan_notes, field_value
                FROM metadata_scan_status 
                WHERE scanned_file_id = ? AND field_name = ?
            ''', (scanned_file_id, field_name))
            result = cursor.fetchone()
            if result:
                return {
                    'field_name': result[0],
                    'scan_status': result[1],
                    'scan_date': result[2],
                    'scan_notes': result[3],
                    'field_value': result[4]
                }
            return None
        else:
            cursor.execute('''
                SELECT field_name, scan_status, scan_date, scan_notes, field_value
                FROM metadata_scan_status 
                WHERE scanned_file_id = ?
            ''', (scanned_file_id,))
            
            results = {}
            for row in cursor.fetchall():
                results[row[0]] = {
                    'scan_status': row[1],
                    'scan_date': row[2], 
                    'scan_notes': row[3],
                    'field_value': row[4]
                }
            return results
    
    def update_metadata_scan_status(self, scanned_file_id: int, field_name: str, 
                                  scan_status: int, field_value: Optional[str] = None, 
                                  scan_notes: Optional[str] = None):
        """Update metadata scan status for a specific field"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO metadata_scan_status 
            (scanned_file_id, field_name, scan_status, scan_date, scan_notes, field_value)
            VALUES (?, ?, ?, strftime('%s', 'now'), ?, ?)
        ''', (scanned_file_id, field_name, scan_status, scan_notes, field_value))
        self.conn.commit()
    
    def get_missing_metadata_fields(self, scanned_file_id: int, required_fields: list) -> list:
        """Get list of metadata fields that haven't been successfully scanned yet"""
        cursor = self.conn.cursor()
        
        # Get already scanned fields
        cursor.execute('''
            SELECT field_name FROM metadata_scan_status 
            WHERE scanned_file_id = ? AND scan_status = 1
        ''', (scanned_file_id,))
        
        scanned_fields = {row[0] for row in cursor.fetchall()}
        missing_fields = [field for field in required_fields if field not in scanned_fields]
        
        return missing_fields
    
    def get_files_needing_specific_metadata_scan(self, file_type: Optional[str] = None, 
                                               missing_fields: Optional[list] = None) -> list:
        """Get files that need metadata scanning for specific fields"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT DISTINCT sf.id, sf.file_path, sf.file_name, sf.file_type
            FROM scanned_files sf
            LEFT JOIN metadata_scan_status mss ON sf.id = mss.scanned_file_id
        '''
        params = []
        
        conditions = []
        if file_type:
            conditions.append("sf.file_type = ?")
            params.append(file_type)
        
        if missing_fields:
            placeholders = ','.join('?' * len(missing_fields))
            conditions.append(f'''
                sf.id NOT IN (
                    SELECT scanned_file_id FROM metadata_scan_status 
                    WHERE field_name IN ({placeholders}) AND scan_status = 1
                )
            ''')
            params.extend(missing_fields)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def is_media_metadata_scanned(self, scanned_file_id: int) -> bool:
        """Check if media metadata has been successfully scanned for this file"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT scan_status FROM metadata_scan_status 
            WHERE scanned_file_id = ? AND field_name = 'image_metadata' AND scan_status = 1
        ''', (scanned_file_id,))
        return cursor.fetchone() is not None

    def mark_media_metadata_scanned(self, scanned_file_id: int, success: bool = True, 
                                   scan_notes: Optional[str] = None):
        """Mark media metadata as scanned for this file"""
        cursor = self.conn.cursor()
        
        scan_status = 1 if success else 2
        field_value = "metadata_extracted" if success else "extraction_failed"
        
        cursor.execute('''
            INSERT OR REPLACE INTO metadata_scan_status 
            (scanned_file_id, field_name, scan_status, scan_date, scan_notes, field_value)
            VALUES (?, ?, ?, strftime('%s', 'now'), ?, ?)
        ''', (scanned_file_id, 'image_metadata', scan_status, scan_notes, field_value))
        
        self.conn.commit()

    def get_unscanned_media_files(self, limit: Optional[int] = None) -> List[Dict]:
        """Get media files that haven't been scanned for metadata yet"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT sf.id, sf.file_path, sf.file_name, sf.file_type 
            FROM scanned_files sf
            LEFT JOIN metadata_scan_status mss ON sf.id = mss.scanned_file_id 
                AND mss.field_name = 'image_metadata' AND mss.scan_status = 1
            WHERE sf.file_type IN ('jpeg', 'jpg', 'png', 'webp', 'gif', 'bmp', 'tiff', 'image', 'video')
            AND mss.scanned_file_id IS NULL
            ORDER BY sf.id
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # Add methods to DatabaseManager class using setattr to avoid type checker issues
    import types
    setattr(DatabaseManager, 'get_metadata_scan_status', get_metadata_scan_status)
    setattr(DatabaseManager, 'update_metadata_scan_status', update_metadata_scan_status)
    setattr(DatabaseManager, 'get_missing_metadata_fields', get_missing_metadata_fields)
    setattr(DatabaseManager, 'is_media_metadata_scanned', is_media_metadata_scanned)
    setattr(DatabaseManager, 'mark_media_metadata_scanned', mark_media_metadata_scanned)
    setattr(DatabaseManager, 'get_unscanned_media_files', get_unscanned_media_files)

# Apply the methods to DatabaseManager
add_metadata_methods_to_database_manager()


class FileScanner:
    """Main file scanner class for generating hashes and detecting model files"""
    
    # Model file extensions (covers all civitai model types)
    MODEL_EXTENSIONS = {
        '.safetensors',    # Modern format (Checkpoints, LoRA, VAE, ControlNet)
        '.ckpt',           # Legacy checkpoints and some LoRA  
        '.pt',             # PyTorch format (LoRA, TextualInversion, Hypernetwork, VAE, Upscaler)
        '.pth',            # PyTorch format variant
        '.bin',            # Binary embeddings (TextualInversion)
        '.vae',            # VAE models (standalone VAE files)
        '.onnx',           # ONNX runtime models
        '.gguf',           # Quantized models (FLUX, etc.)
        '.safe',           # SafeTensors variant/partial files
        '.oldsafetensors'  # Legacy SafeTensors naming
    }
    
    # Associated file extensions
    ASSOCIATED_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.webp',  # Images
        '.civitai.info', '.metadata.json', '.txt', '.yaml', '.yml', '.json'  # Metadata
    }
    
    # Image extensions
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}
    
    # Text/metadata extensions  
    TEXT_EXTENSIONS = {'.txt', '.json', '.yaml', '.yml', '.md', '.info'}
    
    def __init__(self, config_path: str = "config.ini", force_rescan: bool = False, skip_folders: bool = True, verbose: bool = False):
        self.config = self._load_config(config_path)
        self.db: DatabaseManager = DatabaseManager()
        self.scanned_count = 0
        self.hash_cache = {}
        self.progress_callback = None
        self.skip_folders = skip_folders
        self.batch_count = 0
        self.verbose = verbose
        self.batch_size = self.config['database_commit_batch_size']
        self.directory_batch_count = 0
        self.directory_batch_size = self.config['directory_batch_size']
        self.force_rescan = force_rescan
        
        # Track column lacking errors for retry functionality
        self.column_lacking_errors = []
        self.column_lacking_file = 'column_lacking.json'
        
        # Load existing hash cache if available
        cache_file = self.config.get('hash_cache_file', 'model_hashes.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.hash_cache = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load hash cache: {e}")
    
    def _track_column_lacking_error(self, file_path: str, error_msg: str):
        """Track files that failed due to missing database columns for later retry"""
        import re
        import json
        from datetime import datetime
        
        # Extract column name from error message
        column_match = re.search(r'no column named (\w+)', error_msg)
        if column_match:
            column_name = column_match.group(1)
            
            error_entry = {
                'file_path': file_path,
                'column_name': column_name,
                'error_message': error_msg,
                'timestamp': datetime.now().isoformat(),
                'retry_count': 0
            }
            
            self.column_lacking_errors.append(error_entry)
    
    def _save_column_lacking_errors(self):
        """Save column lacking errors to JSON file"""
        if not self.column_lacking_errors:
            return
            
        try:
            import json
            
            # Load existing errors if file exists
            existing_errors = []
            if os.path.exists(self.column_lacking_file):
                try:
                    with open(self.column_lacking_file, 'r') as f:
                        existing_errors = json.load(f)
                except Exception:
                    pass
            
            # Merge with new errors (avoid duplicates based on file_path)
            existing_paths = {error['file_path'] for error in existing_errors}
            new_errors = [error for error in self.column_lacking_errors 
                         if error['file_path'] not in existing_paths]
            
            all_errors = existing_errors + new_errors
            
            # Save updated errors
            with open(self.column_lacking_file, 'w') as f:
                json.dump(all_errors, f, indent=2)
            
            if new_errors:
                print(f"📝 Saved {len(new_errors)} new column lacking errors to {self.column_lacking_file}")
            
        except Exception as e:
            print(f"Warning: Could not save column lacking errors: {e}")
    
    def _load_column_lacking_errors(self) -> list:
        """Load previously failed files due to column lacking errors"""
        try:
            import json
            if os.path.exists(self.column_lacking_file):
                with open(self.column_lacking_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load column lacking errors: {e}")
        return []
    
    def _load_error_log_failures(self) -> list:
        """Load failed files from the error log"""
        failed_files = []
        seen_paths = set()
        
        # Check both possible error log locations
        error_log_paths = [
            "metadata_extractor_errors.json",
            "errors/metadata_extractor_errors.json"
        ]
        
        for error_log_path in error_log_paths:
            if os.path.exists(error_log_path):
                try:
                    import json
                    with open(error_log_path, 'r', encoding='utf-8') as f:
                        errors = json.load(f)
                    
                    # Extract unique file paths that had errors
                    for error in errors:
                        file_path = error.get('file_path')
                        if file_path and file_path not in seen_paths:
                            # Adjust path from server format to local format
                            # Convert /mnt/user/... to /mnt/...
                            if file_path.startswith('/mnt/user/'):
                                file_path = file_path.replace('/mnt/user/', '/mnt/', 1)
                            
                            failed_files.append({
                                'file_path': file_path,
                                'error': error.get('error', 'Unknown error'),
                                'timestamp': error.get('timestamp')
                            })
                            seen_paths.add(file_path)
                
                except Exception as e:
                    print(f"Warning: Could not load error log {error_log_path}: {e}")
        
        return failed_files
    
    def find_missing_files_and_suggest_cleanup(self, limit: int = 100) -> dict:
        """Find files in database that no longer exist at their recorded paths"""
        missing_files = []
        moved_files = []
        
        # Get a sample of files from the database
        files = self.db.get_files_sample(limit)
        
        for file_info in files:
            file_path = file_info['file_path']
            
            # Check if file exists at recorded path
            if not os.path.exists(file_path):
                missing_info = {
                    'id': file_info['id'],
                    'recorded_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'parent_dir': os.path.dirname(file_path)
                }
                
                # Check if parent directory exists
                parent_exists = os.path.exists(missing_info['parent_dir'])
                missing_info['parent_exists'] = parent_exists
                
                if parent_exists:
                    # Check what files are actually in the parent directory
                    try:
                        actual_files = os.listdir(missing_info['parent_dir'])
                        missing_info['actual_files'] = actual_files
                        
                        # Look for similar named files (might have been renamed)
                        base_name = os.path.splitext(missing_info['file_name'])[0]
                        similar_files = [f for f in actual_files if base_name[:10] in f]
                        if similar_files:
                            missing_info['possible_matches'] = similar_files
                            moved_files.append(missing_info)
                        else:
                            missing_files.append(missing_info)
                    except Exception as e:
                        missing_info['directory_error'] = str(e)
                        missing_files.append(missing_info)
                else:
                    missing_files.append(missing_info)
        
        return {
            'missing_files': missing_files,
            'moved_files': moved_files,
            'total_checked': len(files)
        }
    
    def recover_missing_file(self, missing_file_path: str, scanned_file_id: int) -> tuple[bool, str]:
        """Attempt to recover a missing file by checking various recovery strategies
        
        Returns:
            (success: bool, new_path: str or error_message: str)
        """
        
        # Strategy 1: Path translation (/mnt/user/ -> /mnt/)
        if missing_file_path.startswith('/mnt/user/'):
            alt_path = missing_file_path.replace('/mnt/user/', '/mnt/', 1)
            if os.path.exists(alt_path):
                return True, alt_path
        elif missing_file_path.startswith('/mnt/'):
            alt_path = missing_file_path.replace('/mnt/', '/mnt/user/', 1)
            if os.path.exists(alt_path):
                return True, alt_path
        
        # Strategy 2: Check if file was moved to destination directory
        # Get model info if this is an associated file
        filename = os.path.basename(missing_file_path)
        parent_dir = os.path.dirname(missing_file_path)
        
        # Look for model files in the same directory
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT target_path, source_path FROM model_files 
            WHERE source_path LIKE ? AND target_path IS NOT NULL
        ''', (f'{parent_dir}%',))
        
        moved_models = cursor.fetchall()
        
        for target_path, source_path in moved_models:
            if target_path:
                # Check if the missing file exists in the model's target directory
                target_dir = os.path.dirname(target_path)
                potential_location = os.path.join(target_dir, filename)
                if os.path.exists(potential_location):
                    return True, potential_location
        
        # Strategy 3: Rescan the parent directory for similar files
        if os.path.exists(parent_dir):
            try:
                files_in_dir = os.listdir(parent_dir)
                base_name = os.path.splitext(filename)[0]
                extension = os.path.splitext(filename)[1]
                
                # Look for files with similar names
                for file_in_dir in files_in_dir:
                    if (file_in_dir.startswith(base_name[:10]) and 
                        file_in_dir.endswith(extension) and 
                        file_in_dir != filename):
                        potential_path = os.path.join(parent_dir, file_in_dir)
                        if os.path.exists(potential_path):
                            return True, potential_path
                            
            except Exception as e:
                return False, f"Error scanning directory: {e}"
        
        # Strategy 4: Look in common destination directories
        destination_base = self.config.get('destination_directory', '')
        if destination_base:
            # Extract the relative path structure
            relative_path = missing_file_path
            for prefix in ['/mnt/user/', '/mnt/']:
                if relative_path.startswith(prefix):
                    relative_path = relative_path[len(prefix):]
                    break
            
            # Check in destination directory
            potential_dest = os.path.join(destination_base, relative_path)
            if os.path.exists(potential_dest):
                return True, potential_dest
        
        return False, "File not found using any recovery strategy"
    
    def bulk_recover_missing_files(self, dry_run: bool = True, limit: int = 100) -> dict:
        """Attempt to recover multiple missing files in bulk
        
        Args:
            dry_run: If True, only show what would be changed without making changes
            limit: Maximum number of files to check in one run
            
        Returns:
            dict with recovery statistics
        """
        
        # Find files that don't exist at their recorded paths
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT id, file_path FROM scanned_files LIMIT ?', (limit,))
        files_to_check = cursor.fetchall()
        
        results = {
            'total_checked': 0,
            'recovered': 0,
            'still_missing': 0,
            'db_updates': 0,
            'recovery_details': []
        }
        
        for file_id, file_path in files_to_check:
            results['total_checked'] += 1
            
            # Skip if file exists
            if os.path.exists(file_path):
                continue
                
            # Try to recover the file
            success, recovery_result = self.recover_missing_file(file_path, file_id)
            
            if success:
                new_path = recovery_result
                results['recovered'] += 1
                
                recovery_info = {
                    'file_id': file_id,
                    'old_path': file_path,
                    'new_path': new_path,
                    'filename': os.path.basename(file_path)
                }
                results['recovery_details'].append(recovery_info)
                
                print(f"🔄 Recovered: {recovery_info['filename']}")
                print(f"   From: {file_path}")
                print(f"   To:   {new_path}")
                
                if not dry_run:
                    # Update the database
                    cursor.execute('UPDATE scanned_files SET file_path = ? WHERE id = ?', 
                                 (new_path, file_id))
                    results['db_updates'] += 1
                    
            else:
                results['still_missing'] += 1
                if self.config.get('verbose', False):
                    print(f"❌ Could not recover: {os.path.basename(file_path)}")
                    print(f"   Path: {file_path}")
                    print(f"   Reason: {recovery_result}")
        
        if not dry_run and results['db_updates'] > 0:
            self.db.conn.commit()
            print(f"\n✅ Updated {results['db_updates']} file paths in database")
        elif dry_run and results['recovered'] > 0:
            print(f"\n🔍 Dry run: Would update {results['recovered']} file paths")
        
        return results
    
    def migrate_database_paths(self, old_prefix: str, new_prefix: str, dry_run: bool = True) -> dict:
        """Migrate file paths in the database from old prefix to new prefix
        
        Args:
            old_prefix: Old path prefix to replace (e.g., '/mnt/user/')
            new_prefix: New path prefix to use (e.g., '/mnt/')
            dry_run: If True, only show what would be changed without making changes
        """
        # Find files with the old prefix
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT id, file_path FROM scanned_files WHERE file_path LIKE ?', (f'{old_prefix}%',))
        matching_files = cursor.fetchall()
        
        results = {
            'files_found': len(matching_files),
            'files_updated': 0,
            'files_verified': 0,
            'files_missing': 0,
            'changes': []
        }
        
        for file_id, old_path in matching_files:
            # Generate new path
            new_path = old_path.replace(old_prefix, new_prefix, 1)
            
            change_info = {
                'id': file_id,
                'old_path': old_path,
                'new_path': new_path,
                'exists_at_new_path': os.path.exists(new_path),
                'exists_at_old_path': os.path.exists(old_path)
            }
            results['changes'].append(change_info)
            
            if change_info['exists_at_new_path']:
                results['files_verified'] += 1
                if not dry_run:
                    # Update the database
                    cursor.execute('UPDATE scanned_files SET file_path = ? WHERE id = ?', (new_path, file_id))
                    results['files_updated'] += 1
            else:
                results['files_missing'] += 1
        
        if not dry_run:
            self.db.conn.commit()
            print(f"✅ Updated {results['files_updated']} file paths in database")
        else:
            print(f"🔍 Dry run: Would update {results['files_verified']} file paths")
        
        return results
    
    def rescan_and_repair_text_file_paths(self) -> dict:
        """Intelligently scan filesystem and repair database path mismatches for text/metadata files"""
        results = {
            'files_scanned': 0,
            'paths_checked': 0, 
            'paths_corrected': 0,
            'database_updates': 0,
            'errors': 0,
            'error_details': []
        }
        
        # Load destination directory from config.ini
        destination_directory = None
        try:
            import configparser
            config = configparser.ConfigParser()
            config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
            if os.path.exists(config_path):
                config.read(config_path)
                destination_directory = config.get('Paths', 'destination_directory', fallback=None)
                if destination_directory:
                    print(f"📂 Using destination directory from config: {destination_directory}")
            else:
                print(f"⚠️  Config file not found at: {config_path}")
        except Exception as e:
            print(f"⚠️  Error reading config.ini: {e}")
        
        # Check if we have a previous error log to process
        error_log_path = "wrong_path.json"
        previous_errors = []
        if os.path.exists(error_log_path):
            try:
                with open(error_log_path, 'r') as f:
                    previous_errors = json.load(f)
                print(f"📄 Found previous error log with {len(previous_errors)} failed files")
                print(f"   Processing these files first before full scan...")
            except Exception as e:
                print(f"⚠️  Could not load previous error log: {e}")
        
        try:
            cursor = self.db.conn.cursor()
            
            # Get all text-based files from scanned_files that might have path issues
            cursor.execute('''
                SELECT id, file_path, file_name, file_type, extension 
                FROM scanned_files 
                WHERE file_type IN ('text', 'json', 'unknown')
                OR extension IN ('.txt', '.json', '.yaml', '.yml', '.civitai.info', '.metadata.json')
                ORDER BY file_path
            ''')
            
            text_files = cursor.fetchall()
            results['files_scanned'] = len(text_files)
            
            print(f"Found {len(text_files)} text/metadata files to check...")
            print("=" * 80)
            
            # Process previous errors first if they exist
            if previous_errors:
                print(f"\n🔄 PROCESSING PREVIOUS ERRORS FIRST")
                print(f"{'='*80}")
                self._process_error_files(previous_errors, cursor, results, destination_directory)
            
            for file_id, db_path, file_name, file_type, extension in text_files:
                results['paths_checked'] += 1
                
                print(f"\n[{results['paths_checked']:,}/{len(text_files):,}] Checking: {file_name}")
                print(f"  Type: {file_type} | Extension: {extension}")
                print(f"  Database path: {db_path}")
                
                # Check if file exists at recorded path
                if os.path.exists(db_path):
                    print(f"  ✅ File exists at recorded path - OK")
                    continue  # Path is correct, skip
                
                print(f"  ❌ File NOT found at recorded path")
                
                # Try to find the file using comprehensive search
                found_path = self._find_file_comprehensive(db_path, file_name, destination_directory)
                
                if found_path:
                    # Check if target path already exists in database (duplicate entry)
                    cursor.execute('SELECT id, file_path FROM scanned_files WHERE file_path = ?', (found_path,))
                    existing_record = cursor.fetchone()
                    
                    if existing_record and existing_record[0] != file_id:
                        print(f"  🔄 DUPLICATE DETECTED:")
                        print(f"     CURRENT RECORD (ID {file_id}): {db_path}")
                        print(f"     EXISTING RECORD (ID {existing_record[0]}): {found_path}")
                        print(f"  🗑️  REMOVING DUPLICATE RECORD (keeping the one with correct path)")
                        
                        # Remove the current record (incorrect path) since the correct one already exists
                        cursor.execute('DELETE FROM scanned_files WHERE id = ?', (file_id,))
                        results['paths_corrected'] += 1
                        results['database_updates'] += 1
                        print(f"  ✅ Duplicate record removed successfully")
                    else:
                        # Update the database with the correct path
                        print(f"  🔄 CORRECTING DATABASE PATH:")
                        print(f"     FROM: {db_path}")
                        print(f"     TO:   {found_path}")
                        
                        cursor.execute('''
                            UPDATE scanned_files 
                            SET file_path = ?, updated_at = strftime('%s', 'now')
                            WHERE id = ?
                        ''', (found_path, file_id))
                        
                        results['paths_corrected'] += 1
                        results['database_updates'] += 1
                        print(f"  ✅ Database updated successfully")
                else:
                    # File not found at any expected location
                    print(f"  ⚠️  FILE MISSING: Cannot find {file_name} at any expected location")
                    error_detail = {
                        'file_id': file_id,
                        'file_name': file_name,
                        'file_type': file_type,
                        'extension': extension,
                        'database_path': db_path,
                        'table': 'scanned_files',
                        'error_type': 'file_not_found'
                    }
                    results['error_details'].append(error_detail)
                    results['errors'] += 1
                
                # Progress summary every 100 files
                if results['paths_checked'] % 100 == 0:
                    print(f"\n{'='*60}")
                    print(f"PROGRESS SUMMARY: {results['paths_checked']:,}/{len(text_files):,} files checked")
                    print(f"  ✅ Paths corrected: {results['paths_corrected']}")
                    print(f"  ⚠️  Files missing: {results['errors']}")
                    print(f"{'='*60}")
            
            # Also check associated_files table for path mismatches
            cursor.execute('''
                SELECT id, source_path, target_path, association_type 
                FROM associated_files 
                ORDER BY source_path
            ''')
            
            assoc_files = cursor.fetchall()
            print(f"\n{'='*80}")
            print(f"CHECKING ASSOCIATED FILES TABLE")
            print(f"Found {len(assoc_files)} associated file records to check...")
            print(f"{'='*80}")
            
            assoc_count = 0
            for assoc_id, source_path, target_path, assoc_type in assoc_files:
                assoc_count += 1
                results['paths_checked'] += 1
                
                file_name = os.path.basename(source_path)
                print(f"\n[{assoc_count:,}/{len(assoc_files):,}] Associated File: {file_name}")
                print(f"  Association type: {assoc_type}")
                print(f"  Source path: {source_path}")
                print(f"  Target path: {target_path}")
                
                # Check both source and target paths
                source_found = False
                target_found = False
                
                if source_path and os.path.exists(source_path):
                    print(f"  ✅ Source path exists - OK")
                    source_found = True
                else:
                    print(f"  ❌ Source path NOT found")
                
                if target_path and os.path.exists(target_path):
                    print(f"  ✅ Target path exists - OK")
                    target_found = True
                elif target_path:
                    print(f"  ❌ Target path NOT found")
                
                # Try to fix paths if needed
                corrected_source = None
                corrected_target = None
                
                if not source_found and source_path:
                    corrected_source = self._find_file_comprehensive(source_path, file_name, destination_directory)
                    if corrected_source:
                        print(f"  🔍 Found corrected source: {corrected_source}")
                
                if not target_found and target_path:
                    corrected_target = self._find_file_comprehensive(target_path, file_name, destination_directory)
                    if corrected_target:
                        print(f"  🔍 Found corrected target: {corrected_target}")
                
                # Update database if we found corrections
                if corrected_source or corrected_target:
                    new_source = corrected_source if corrected_source else source_path
                    new_target = corrected_target if corrected_target else target_path
                    
                    print(f"  🔄 CORRECTING ASSOCIATED FILE PATHS:")
                    if corrected_source:
                        print(f"     SOURCE FROM: {source_path}")
                        print(f"     SOURCE TO:   {new_source}")
                    if corrected_target:
                        print(f"     TARGET FROM: {target_path}")
                        print(f"     TARGET TO:   {new_target}")
                    
                    cursor.execute('''
                        UPDATE associated_files 
                        SET source_path = ?, target_path = ?
                        WHERE id = ?
                    ''', (new_source, new_target, assoc_id))
                    
                    results['paths_corrected'] += 1
                    results['database_updates'] += 1
                    print(f"  ✅ Database updated successfully")
                elif not source_found or (target_path and not target_found):
                    print(f"  ⚠️  ASSOCIATED FILE MISSING: {file_name}")
                    error_detail = {
                        'assoc_id': assoc_id,
                        'file_name': file_name,
                        'association_type': assoc_type,
                        'source_path': source_path,
                        'target_path': target_path,
                        'table': 'associated_files',
                        'error_type': 'file_not_found',
                        'source_found': source_found,
                        'target_found': target_found
                    }
                    results['error_details'].append(error_detail)
                    results['errors'] += 1
                
                # Progress summary every 50 associated files
                if assoc_count % 50 == 0:
                    print(f"\n{'='*60}")
                    print(f"ASSOCIATED FILES PROGRESS: {assoc_count:,}/{len(assoc_files):,} checked")
                    print(f"  ✅ Paths corrected: {results['paths_corrected']}")
                    print(f"  ⚠️  Files missing: {results['errors']}")
                    print(f"{'='*60}")
            
            # Save error details to JSON file for next run
            if results['error_details']:
                print(f"\n💾 SAVING ERROR LOG")
                print(f"{'='*60}")
                with open(error_log_path, 'w') as f:
                    json.dump(results['error_details'], f, indent=2)
                print(f"✅ Saved {len(results['error_details'])} errors to: {error_log_path}")
                print(f"   Run this command again to focus on these errors only")
            elif os.path.exists(error_log_path):
                # Remove old error log if no errors found
                os.remove(error_log_path)
                print(f"✅ Removed old error log - all paths now correct!")
            
            # Commit all changes
            if results['database_updates'] > 0:
                self.db.conn.commit()
                print(f"\n{'='*80}")
                print("COMMITTING DATABASE CHANGES")
                print(f"{'='*80}")
                print(f"✅ Successfully committed {results['database_updates']} database updates to disk")
            else:
                print(f"\n{'='*80}")
                print("NO DATABASE CHANGES NEEDED")
                print(f"{'='*80}")
                print("✅ All paths were already correct!")
            
            # Final comprehensive report
            print(f"\n{'='*80}")
            print("FINAL SCAN AND REPAIR REPORT")
            print(f"{'='*80}")
            print(f"📊 STATISTICS:")
            print(f"   • Total files scanned: {results['files_scanned']:,}")
            print(f"   • Total paths checked: {results['paths_checked']:,}")
            print(f"   • Paths corrected: {results['paths_corrected']:,}")
            print(f"   • Database updates: {results['database_updates']:,}")
            print(f"   • Files missing: {results['errors']:,}")
            
            if results['paths_corrected'] > 0:
                print(f"\n✅ SUCCESS: Fixed {results['paths_corrected']} path mismatches")
                print("   The following path corrections were made:")
                print("   • Database paths updated to match actual file locations")
                print("   • Mount point mismatches resolved (/mnt/user/ ↔ /mnt/)")
                print("   • Checked both source AND target paths")
                print("   • Used comprehensive search strategies")
            
            if results['errors'] > 0:
                print(f"\n⚠️  WARNING: {results['errors']} files could not be located")
                print("   These files may have been:")
                print("   • Moved to a different location")
                print("   • Deleted from the filesystem") 
                print("   • Located on an unmounted drive")
                print(f"\n💾 ERROR LOG: Saved details to 'wrong_path.json'")
                print("   • Run this command again to focus only on these errors")
                print("   • Manual investigation may be needed for remaining files")
            
            if results['paths_corrected'] == 0 and results['errors'] == 0:
                print(f"\n✅ PERFECT: All paths are already correct!")
                print("   No path corrections were needed.")
            
            print(f"\n{'='*80}")
            print("ENHANCED FEATURES:")
            print("🔍 Comprehensive search strategies:")
            print("   1. Mount point alternatives (/mnt/user/ ↔ /mnt/)")
            print("   2. Target/sorted location checking")
            print("   3. Nearby directory searching")
            print("📄 Error logging for quick re-runs")
            print("🎯 Both source AND target path validation")
            print(f"{'='*80}")
            
            return results
            
        except Exception as e:
            print(f"Error in path repair: {e}")
            results['errors'] += 1
            return results

    def enhance_and_repair_model_records(self) -> dict:
        """Enhance model records with civitai data, pattern-based base model detection, and path verification"""
        print("🔧 Enhancing model records with comprehensive detection...")
        
        results = {
            'models_checked': 0,
            'paths_corrected': 0,
            'models_enhanced_civitai': 0,
            'base_models_detected': 0,
            'models_missing': 0,
            'errors': 0
        }
        
        unmatched_models = []  # Collect models that couldn't be pattern-matched
        
        try:
            # Import ModelSorter for enhancement capabilities
            from model_sorter import ModelSorter
            sorter = ModelSorter()
            
            # Get all model records from model_sorter database (local copy)
            model_sorter_db_path = os.path.join(os.path.dirname(__file__), 'model_sorter.sqlite')
            if not os.path.exists(model_sorter_db_path):
                print(f"❌ Model sorter database not found at: {model_sorter_db_path}")
                return results
            
            model_conn = sqlite3.connect(model_sorter_db_path)
            model_conn.row_factory = sqlite3.Row
            model_cursor = model_conn.cursor()
            
            # Get all model files with their scanned file information
            model_cursor.execute('''
                SELECT 
                    mf.id,
                    sf.file_name,
                    sf.sha256,
                    mf.source_path,
                    mf.target_path,
                    mf.model_name,
                    mf.base_model
                FROM model_files mf
                JOIN scanned_files sf ON mf.scanned_file_id = sf.id
                ORDER BY mf.id
            ''')
            
            model_records = model_cursor.fetchall()
            total_models = len(model_records)
            print(f"Found {total_models} model records to enhance")
            
            for i, record in enumerate(model_records, 1):
                if i % 100 == 0:
                    print(f"Progress: {i}/{total_models} models processed")
                
                results['models_checked'] += 1
                model_id = record['id']
                current_source_path = record['source_path']
                current_target_path = record['target_path']
                current_base_model = record['base_model']
                current_model_name = record['model_name']
                
                # Step 1: Verify and repair paths (similar to text file repair)
                path_corrected = False
                actual_source_path = current_source_path
                
                if current_source_path and not os.path.exists(current_source_path):
                    # Try path variants
                    path_variants = [
                        current_source_path.replace('/mnt/user/', '/mnt/'),
                        current_source_path.replace('/mnt/', '/mnt/user/'),
                    ]
                    
                    for variant_path in path_variants:
                        if os.path.exists(variant_path):
                            print(f"📍 Correcting model path: {os.path.basename(current_source_path)}")
                            print(f"   From: {current_source_path}")
                            print(f"   To:   {variant_path}")
                            
                            model_cursor.execute('''
                                UPDATE model_files 
                                SET source_path = ?
                                WHERE id = ?
                            ''', (variant_path, model_id))
                            
                            actual_source_path = variant_path
                            path_corrected = True
                            results['paths_corrected'] += 1
                            break
                
                # Step 2: Create model dictionary for processing
                model_dict = {
                    'id': model_id,
                    'file_name': record['file_name'],
                    'source_path': actual_source_path,
                    'target_path': current_target_path,
                    'model_name': current_model_name,
                    'base_model': current_base_model,
                    'sha256': record['sha256']
                }
                
                file_found = False
                if actual_source_path and os.path.exists(actual_source_path):
                    file_found = True
                elif actual_source_path:
                    results['models_missing'] += 1
                    
                    # Check if file exists at target path (where it might have been moved)
                    if current_target_path and os.path.exists(current_target_path):
                        print(f"🎯 Found model at target location: {os.path.basename(current_target_path)}")
                        actual_source_path = current_target_path  # Use target path for processing
                        model_dict['source_path'] = actual_source_path
                        file_found = True
                    else:
                        # Try to construct target path from config if not in database
                        try:
                            base_model = current_base_model or 'Unknown'
                            model_name = current_model_name or record['file_name'].replace('.safetensors', '').replace('.ckpt', '').replace('.pt', '')
                            file_name = record['file_name']
                            
                            # Try common destination patterns
                            possible_targets = [
                                f"{sorter.config.get('main', 'destination_path')}/loras/{base_model}/{model_name}/{file_name}",
                                f"{sorter.config.get('main', 'destination_path')}/checkpoints/{base_model}/{model_name}/{file_name}",
                                f"{sorter.config.get('main', 'destination_path')}/vae/{base_model}/{model_name}/{file_name}",
                                f"{sorter.config.get('main', 'destination_path')}/hypernetwork/{base_model}/{model_name}/{file_name}",
                            ]
                            
                            for target_path in possible_targets:
                                if os.path.exists(target_path):
                                    actual_source_path = target_path
                                    model_dict['source_path'] = actual_source_path
                                    file_found = True
                                    if self.config['verbose']:
                                        print(f"🎯 Found model at constructed target location: {os.path.basename(actual_source_path)}")
                                    break
                        except Exception as e:
                            if self.config['verbose']:
                                print(f"   Could not construct target path: {e}")
                
                if not file_found and actual_source_path:
                    # Still process for pattern detection even if file is missing
                    # since we can detect base model from path patterns
                    print(f"⚠️ Model file missing, but processing for pattern detection: {os.path.basename(actual_source_path)}")
                
                # Track original values
                original_model_name = current_model_name
                original_base_model = current_base_model
                
                # Enhance with civitai and pattern detection (regardless of file existence)
                enhanced_model = sorter.enhance_model_info_from_civitai(model_dict)
                
                # Check what was enhanced
                civitai_enhanced = enhanced_model.get('model_name') != original_model_name
                base_model_detected = enhanced_model.get('base_model') != original_base_model
                
                if civitai_enhanced:
                    results['models_enhanced_civitai'] += 1
                
                if base_model_detected and enhanced_model.get('base_model') != 'Unknown':
                    results['base_models_detected'] += 1
                
                # Collect unmatched models for analysis
                if enhanced_model.get('base_model') == 'Unknown' or enhanced_model.get('base_model') == original_base_model == 'Unknown':
                    unmatched_models.append(model_dict)
                
                # Update database if anything changed
                if (enhanced_model.get('model_name') != original_model_name or 
                    enhanced_model.get('base_model') != original_base_model or 
                    path_corrected):
                    
                    model_cursor.execute('''
                        UPDATE model_files 
                        SET model_name = ?, base_model = ?
                        WHERE id = ?
                    ''', (
                        enhanced_model.get('model_name', original_model_name),
                        enhanced_model.get('base_model', original_base_model),
                        model_id
                    ))
            
            model_conn.commit()
            model_conn.close()
            
            # Log unmatched models for analysis
            sorter.log_unmatched_models(unmatched_models)
            
            sorter.close()
            
        except Exception as e:
            print(f"❌ Error enhancing models: {e}")
            results['errors'] += 1
            import traceback
            traceback.print_exc()
        
        return results

    def enhance_models_direct(self, collection_db_path: str, apply_updates: bool = False, verbose: bool = False) -> dict:
        """Directly enhance Unknown/Other models from collection database with batch updates"""
        print(f"🎯 Directly enhancing Unknown/Other models from: {collection_db_path}")
        
        results = {
            'unknown_models_checked': 0,
            'other_models_checked': 0,
            'models_enhanced_civitai': 0,
            'base_models_detected': 0,
            'base_models_detected_files': 0,
            'models_missing': 0,
            'models_updated': 0,
            'errors': 0
        }
        
        unmatched_models = []
        pending_updates = []
        
        try:
            # Import ModelSorter for enhancement capabilities
            from model_sorter import ModelSorter
            sorter = ModelSorter()
            
            # Check if collection database exists
            if not os.path.exists(collection_db_path):
                print(f"❌ Collection database not found at: {collection_db_path}")
                return results
            
            import sqlite3
            import time
            
            # Step 1: Read Unknown/Other models from collection database (single read)
            print(f"📖 Reading Unknown/Other models from collection database...")
            collection_conn = sqlite3.connect(collection_db_path, timeout=30.0)
            collection_conn.row_factory = sqlite3.Row
            collection_cursor = collection_conn.cursor()
            
            # Get Unknown and Other models in one query
            collection_cursor.execute('''
                SELECT 
                    mf.id,
                    sf.file_name,
                    sf.sha256,
                    mf.source_path,
                    mf.target_path,
                    mf.model_name,
                    mf.base_model
                FROM model_files mf
                JOIN scanned_files sf ON mf.scanned_file_id = sf.id
                WHERE mf.base_model IN ('Unknown', 'Other')
                ORDER BY mf.base_model, mf.id
            ''')
            
            model_records = collection_cursor.fetchall()
            collection_conn.close()  # Close immediately after reading
            
            total_models = len(model_records)
            print(f"Found {total_models} Unknown/Other models to enhance")
            
            if total_models == 0:
                print("✅ No Unknown/Other models found - all models are already categorized!")
                sorter.close()
                return results
            
            print(f"� Processing {total_models} models with batch updates every 100 models...")
            
            # Step 2: Process each model (in memory)
            for i, record in enumerate(model_records, 1):
                if i % 100 == 0:
                    print(f"Progress: {i}/{total_models} models processed")
                
                model_id = record['id']
                current_base_model = record['base_model']
                
                if current_base_model == 'Unknown':
                    results['unknown_models_checked'] += 1
                elif current_base_model == 'Other':
                    results['other_models_checked'] += 1
                
                # Create model dictionary for processing
                model_dict = {
                    'id': model_id,
                    'file_name': record['file_name'],
                    'source_path': record['source_path'],
                    'target_path': record['target_path'],
                    'model_name': record['model_name'],
                    'base_model': current_base_model,
                    'sha256': record['sha256']
                }
                
                # Check if file exists
                actual_source_path = record['source_path']
                file_found = False
                
                if actual_source_path and os.path.exists(actual_source_path):
                    file_found = True
                elif actual_source_path:
                    results['models_missing'] += 1
                    
                    # Try target path first
                    if record['target_path'] and os.path.exists(record['target_path']):
                        actual_source_path = record['target_path']
                        model_dict['source_path'] = actual_source_path
                        file_found = True
                        if verbose:
                            print(f"🎯 Found model at target location: {os.path.basename(actual_source_path)}")
                    else:
                        # Try to construct target path from config if not in database
                        try:
                            base_model = record['base_model'] or 'Unknown'
                            model_name = record['model_name'] or record['file_name'].replace('.safetensors', '').replace('.ckpt', '').replace('.pt', '')
                            file_name = record['file_name']
                            
                            # Try common destination patterns
                            possible_targets = [
                                f"{sorter.config.get('main', 'destination_path')}/loras/{base_model}/{model_name}/{file_name}",
                                f"{sorter.config.get('main', 'destination_path')}/checkpoints/{base_model}/{model_name}/{file_name}",
                                f"{sorter.config.get('main', 'destination_path')}/vae/{base_model}/{model_name}/{file_name}",
                                f"{sorter.config.get('main', 'destination_path')}/hypernetwork/{base_model}/{model_name}/{file_name}",
                            ]
                            
                            for target_path in possible_targets:
                                if os.path.exists(target_path):
                                    actual_source_path = target_path
                                    model_dict['source_path'] = actual_source_path
                                    file_found = True
                                    if verbose:
                                        print(f"🎯 Found model at constructed target location: {os.path.basename(actual_source_path)}")
                                    break
                        except Exception as e:
                            if verbose:
                                print(f"   Could not construct target path: {e}")
                
                if not file_found and actual_source_path:
                    print(f"⚠️ Model file missing, processing for pattern detection: {os.path.basename(actual_source_path)}")
                
                # Track original values
                original_model_name = record['model_name']
                original_base_model = current_base_model
                
                # Enhance with civitai and pattern detection
                enhanced_model = sorter.enhance_model_info_from_civitai(model_dict)
                
                # Check what was enhanced
                civitai_enhanced = enhanced_model.get('model_name') != original_model_name
                base_model_detected = enhanced_model.get('base_model') != original_base_model
                
                if civitai_enhanced:
                    results['models_enhanced_civitai'] += 1
                
                if base_model_detected and enhanced_model.get('base_model') not in ['Unknown', 'Other']:
                    if 'detected base model from associated files' in str(enhanced_model):
                        results['base_models_detected_files'] += 1
                    else:
                        results['base_models_detected'] += 1
                
                # Collect unmatched models for analysis
                if enhanced_model.get('base_model') in ['Unknown', 'Other']:
                    unmatched_models.append(model_dict)
                
                # Prepare update if anything changed
                if (enhanced_model.get('model_name') != original_model_name or 
                    enhanced_model.get('base_model') != original_base_model):
                    
                    pending_updates.append({
                        'id': model_id,
                        'model_name': enhanced_model.get('model_name', original_model_name),
                        'base_model': enhanced_model.get('base_model', original_base_model)
                    })
                
                # Batch update every 100 models
                if len(pending_updates) >= 100:
                    self._apply_batch_updates(collection_db_path, pending_updates)
                    results['models_updated'] += len(pending_updates)
                    pending_updates.clear()
                    print(f"💾 Applied batch update for models up to #{i}")
            
            # Apply remaining updates
            if pending_updates:
                self._apply_batch_updates(collection_db_path, pending_updates)
                results['models_updated'] += len(pending_updates)
                print(f"💾 Applied final batch update for {len(pending_updates)} models")
            
            # Log unmatched models for analysis
            sorter.log_unmatched_models(unmatched_models)
            sorter.close()
            
        except Exception as e:
            print(f"❌ Error in direct model enhancement: {e}")
            results['errors'] += 1
            import traceback
            traceback.print_exc()
        
        return results

    def sort_models_batch(self) -> dict:
        """Sort models in batches for improved performance"""
        batch_size = self.config.get('model_sorting_batch_size', 100)
        print(f"🚀 Starting batch model sorting with batch size: {batch_size}")
        
        results = {
            'models_processed': 0,
            'batches_completed': 0,
            'models_moved': 0,
            'files_moved': 0,
            'database_updates': 0,
            'errors': 0
        }
        
        try:
            # Import ModelSorter for sorting capabilities
            from model_sorter import ModelSorter
            sorter = ModelSorter()
            
            # Get all models that need sorting
            print("📊 Getting models to sort...")
            all_models = sorter.get_models_to_sort()
            total_models = len(all_models)
            
            if total_models == 0:
                print("✅ No models need sorting!")
                sorter.close()
                return results
            
            print(f"Found {total_models} models to sort")
            print(f"Processing in batches of {batch_size} models...")
            
            # Process models in batches
            for batch_start in range(0, total_models, batch_size):
                batch_end = min(batch_start + batch_size, total_models)
                batch_models = all_models[batch_start:batch_end]
                batch_number = (batch_start // batch_size) + 1
                total_batches = (total_models + batch_size - 1) // batch_size
                
                print(f"\n🔄 Processing batch {batch_number}/{total_batches} (models {batch_start + 1}-{batch_end})")
                
                # Step 1: Pre-load all database info for this batch into memory
                print(f"  📋 Pre-loading database info for {len(batch_models)} models...")
                batch_with_enhanced_info = self._preload_batch_database_info(sorter, batch_models)
                
                # Step 2: Move all files in this batch (using pre-loaded info, no DB queries)
                batch_updates = []
                batch_associations = []
                
                for i, enhanced_model in enumerate(batch_with_enhanced_info):
                    model_progress = batch_start + i + 1
                    model_name = enhanced_model['model_name'] or 'None'
                    base_model = enhanced_model['base_model'] or 'Unknown'
                    
                    print(f"  [{model_progress}/{total_models}] Moving: {model_name} ({base_model})")
                    
                    try:
                        # Move the model using pre-loaded info (no database queries)
                        move_result = self._move_model_with_preloaded_info(sorter, enhanced_model)
                        
                        if move_result['success']:
                            results['models_moved'] += 1
                            results['files_moved'] += move_result['files_moved']
                            
                            # Collect database updates for batch processing
                            if move_result['database_update']:
                                batch_updates.append(move_result['database_update'])
                            
                            # Collect association records for batch processing
                            batch_associations.extend(move_result['associations'])
                        
                    except Exception as e:
                        print(f"    ❌ Error moving model {model_name}: {e}")
                        results['errors'] += 1
                
                # Step 2: Batch update database for this batch
                if batch_updates or batch_associations:
                    print(f"  💾 Updating database for batch {batch_number} ({len(batch_updates)} model updates, {len(batch_associations)} associations)")
                    
                    try:
                        self._apply_batch_database_updates(batch_updates, batch_associations)
                        results['database_updates'] += len(batch_updates) + len(batch_associations)
                    except Exception as e:
                        print(f"    ❌ Error updating database for batch {batch_number}: {e}")
                        results['errors'] += 1
                
                results['models_processed'] += len(batch_models)
                results['batches_completed'] += 1
                
                print(f"  ✅ Batch {batch_number} completed - {len(batch_models)} models processed")
            
            sorter.close()
            
        except Exception as e:
            print(f"❌ Error in batch model sorting: {e}")
            results['errors'] += 1
            import traceback
            traceback.print_exc()
        
        return results
    
    def _preload_batch_database_info(self, sorter, batch_models: list) -> list:
        """Pre-load all database information needed for a batch of models to avoid individual DB queries"""
        enhanced_models = []
        
        try:
            # Get all model IDs for this batch
            model_ids = [model['id'] for model in batch_models]
            
            # Pre-load civitai database info for base model enhancement
            print("  🔍 Enhancing model information using civitai database...")
            civitai_enhancements = self._bulk_enhance_models_from_civitai(model_ids)
            
            # Pre-load associated files info for each model
            print("  📎 Pre-loading associated files information...")
            associated_files_cache = {}
            for model in batch_models:
                source_path = model['source_path']
                model_name = model['model_name'] or 'Unknown'
                if os.path.exists(source_path):
                    associated_files_cache[model['id']] = sorter.get_associated_files_for_model(source_path, model_name)
                else:
                    associated_files_cache[model['id']] = []
            
            # Create enhanced model objects with all pre-loaded info
            for i, model in enumerate(batch_models):
                enhanced_model = model.copy()
                
                # Apply civitai enhancements if available
                if model['id'] in civitai_enhancements:
                    enhancement = civitai_enhancements[model['id']]
                    enhanced_model['base_model'] = enhancement.get('base_model', model['base_model'])
                    enhanced_model['model_type'] = enhancement.get('model_type', model['model_type'])
                
                # Add pre-loaded associated files
                enhanced_model['preloaded_associated_files'] = associated_files_cache.get(model['id'], [])
                
                enhanced_models.append(enhanced_model)
            
            return enhanced_models
            
        except Exception as e:
            print(f"    ⚠️  Error pre-loading batch info: {e}")
            # Fallback to original models if pre-loading fails
            return batch_models
    
    def _bulk_enhance_models_from_civitai(self, model_ids: list) -> dict:
        """Bulk enhance models using civitai database to avoid individual queries"""
        enhancements = {}
        
        try:
            civitai_db_path = self.config.get('database_path', 'Database/civitai.sqlite')
            if not os.path.exists(civitai_db_path):
                return enhancements
            
            import sqlite3
            conn = sqlite3.connect(civitai_db_path, timeout=30.0)
            cursor = conn.cursor()
            
            # Get current model_sorter database path
            local_db_path = os.path.join(os.path.dirname(__file__), 'model_sorter.sqlite')
            if not os.path.exists(local_db_path):
                conn.close()
                return enhancements
            
            local_conn = sqlite3.connect(local_db_path, timeout=30.0)
            local_cursor = local_conn.cursor()
            
            # Bulk query model info with hash from scanned_files
            model_ids_str = ','.join(['?' for _ in model_ids])
            local_cursor.execute(f'''
                SELECT mf.id, sf.blake3, mf.model_name, sf.file_name
                FROM model_files mf
                JOIN scanned_files sf ON mf.scanned_file_id = sf.id
                WHERE mf.id IN ({model_ids_str})
            ''', model_ids)
            
            model_info = {row[0]: {'blake3': row[1], 'model_name': row[2], 'file_name': row[3]} 
                         for row in local_cursor.fetchall()}
            
            # Bulk query civitai database for enhancements
            for model_id, info in model_info.items():
                blake3 = info['blake3']
                model_name = info['model_name']
                file_name = info['file_name']
                
                if blake3:
                    # Try hash lookup first
                    cursor.execute('''
                        SELECT DISTINCT m.name, m.type, mv.baseModel 
                        FROM models m
                        JOIN modelVersions mv ON m.id = mv.modelId
                        JOIN files f ON mv.id = f.modelVersionId
                        WHERE f.hashes_BLAKE3 = ? OR f.hashes_SHA256 = ?
                        LIMIT 1
                    ''', (blake3, blake3))
                    
                    result = cursor.fetchone()
                    if result:
                        enhancements[model_id] = {
                            'base_model': result[2] or 'Unknown',
                            'model_type': result[1] or 'LORA'
                        }
                        continue
                
                # Fallback to pattern-based detection if no hash match
                if model_name or file_name:
                    search_name = model_name or file_name or ''
                    detected_base = self._detect_base_model_from_patterns(search_name)
                    if detected_base != 'Unknown':
                        enhancements[model_id] = {
                            'base_model': detected_base,
                            'model_type': 'LORA'  # Default assumption
                        }
            
            local_conn.close()
            conn.close()
            
        except Exception as e:
            print(f"    ⚠️  Error in bulk civitai enhancement: {e}")
        
        return enhancements
    
    def _detect_base_model_from_patterns(self, text: str) -> str:
        """Quick pattern-based base model detection"""
        if not text:
            return 'Unknown'
        
        text_lower = text.lower()
        
        # Quick pattern matching for common base models
        if any(pattern in text_lower for pattern in ['pony', 'ponydiffusion']):
            return 'Pony'
        elif any(pattern in text_lower for pattern in ['sdxl', 'xl']):
            return 'SDXL 1.0'
        elif any(pattern in text_lower for pattern in ['sd15', 'sd 1.5', 'sd_1_5']):
            return 'SD 1.5'
        elif any(pattern in text_lower for pattern in ['flux', 'flux1', 'flux.1']):
            return 'Flux.1 D'
        elif any(pattern in text_lower for pattern in ['wan2.2', 'wan 2.2', 'wan_2_2']):
            return 'Wan Video 2.2 T2V-A14B'
        else:
            return 'Unknown'
    
    def _move_model_with_preloaded_info(self, sorter, enhanced_model: dict) -> dict:
        """Move a model using pre-loaded database info to avoid DB queries during file operations"""
        result = {
            'success': False,
            'files_moved': 0,
            'database_update': None,
            'associations': []
        }
        
        try:
            source_path = enhanced_model['source_path']
            model_name = enhanced_model['model_name'] or 'None'
            
            # Determine target structure using pre-loaded model info
            target_directory, folder_path = sorter.determine_model_folder_structure(enhanced_model)
            model_filename = os.path.basename(source_path)
            target_model_path = os.path.join(target_directory, model_filename)
            
            # Check if source file exists, or if it's already been moved to target
            if not os.path.exists(source_path):
                # Check if file already exists at target destination
                if os.path.exists(target_model_path):
                    print(f"    ✅ Model already at destination: {target_model_path}")
                    final_path = target_model_path
                    result['files_moved'] = 0  # Already moved, don't count as new move
                    
                    # Still need to update database with correct target path
                    result['database_update'] = {
                        'model_id': enhanced_model['id'],
                        'target_path': final_path,
                        'status': 'moved',
                        'model_name': enhanced_model['model_name'],
                        'base_model': enhanced_model['base_model'],
                        'scanned_file_id': enhanced_model.get('scanned_file_id'),
                        'model_type': enhanced_model.get('model_type', 'LORA')
                    }
                    result['success'] = True
                else:
                    print(f"    ⚠️  Source file missing: {source_path}")
                    return result
            else:
                # File exists at source, move it normally
                success, final_path = sorter.file_mover.move_file(source_path, target_model_path)
                if not success:
                    return result
                
                result['files_moved'] += 1
                print(f"    Moved: {source_path} -> {final_path}")
                
                # Prepare database update for the main model
                result['database_update'] = {
                    'model_id': enhanced_model['id'],
                    'target_path': final_path,
                    'status': 'moved',
                    'model_name': enhanced_model['model_name'],
                    'base_model': enhanced_model['base_model'],
                    'scanned_file_id': enhanced_model.get('scanned_file_id'),
                    'model_type': enhanced_model.get('model_type', 'LORA')
                }
            
            # Use pre-loaded associated files (no database queries needed)
            associated_files = enhanced_model.get('preloaded_associated_files', [])
            
            for assoc_file in associated_files:
                assoc_filename = os.path.basename(assoc_file)
                assoc_target_path = os.path.join(target_directory, assoc_filename)
                
                # Check if associated file exists at source or is already at destination
                if os.path.exists(assoc_file):
                    # File exists at source, move it
                    assoc_success, assoc_final_path = sorter.file_mover.move_file(assoc_file, assoc_target_path)
                    if assoc_success:
                        result['files_moved'] += 1
                        print(f"    Moved: {assoc_file} -> {assoc_final_path}")
                        
                        # Track association for database update
                        result['associations'].append({
                            'model_id': enhanced_model['id'],
                            'source_path': assoc_file,
                            'target_path': assoc_final_path,
                            'file_name': assoc_filename
                        })
                elif os.path.exists(assoc_target_path):
                    # Associated file already exists at target destination
                    print(f"    ✅ Associated file already at destination: {assoc_target_path}")
                    
                    # Still track association for database update
                    result['associations'].append({
                        'model_id': enhanced_model['id'],
                        'source_path': assoc_file,
                        'target_path': assoc_target_path,
                        'file_name': assoc_filename
                    })
            
            result['success'] = True
            
        except Exception as e:
            print(f"    ❌ Error in _move_model_with_preloaded_info: {e}")
        
        return result
    
    def _move_model_with_tracking(self, sorter, model: dict) -> dict:
        """Move a model and track all changes for batch database updates"""
        result = {
            'success': False,
            'files_moved': 0,
            'database_update': None,
            'associations': []
        }
        
        try:
            source_path = model['source_path']
            model_name = model['model_name'] or 'None'
            
            # Check if source file exists
            if not os.path.exists(source_path):
                print(f"    ⚠️  Source file missing: {source_path}")
                return result
            
            # Determine target structure using ModelSorter logic
            target_directory, folder_path = sorter.determine_model_folder_structure(model)
            model_filename = os.path.basename(source_path)
            target_model_path = os.path.join(target_directory, model_filename)
            
            # Move the main model file
            success, final_path = sorter.file_mover.move_file(source_path, target_model_path)
            if not success:
                return result
            
            result['files_moved'] += 1
            print(f"    Moved: {source_path} -> {final_path}")
            
            # Prepare database update for the main model
            result['database_update'] = {
                'model_id': model['id'],
                'target_path': final_path,
                'status': 'moved',
                'model_name': model['model_name'],
                'base_model': model['base_model']
            }
            
            # Find and move associated files
            associated_files = sorter.get_associated_files_for_model(source_path, model_name)
            
            for assoc_file in associated_files:
                if os.path.exists(assoc_file):
                    assoc_filename = os.path.basename(assoc_file)
                    assoc_target_path = os.path.join(target_directory, assoc_filename)
                    
                    assoc_success, assoc_final_path = sorter.file_mover.move_file(assoc_file, assoc_target_path)
                    if assoc_success:
                        result['files_moved'] += 1
                        print(f"    Moved: {assoc_file} -> {assoc_final_path}")
                        
                        # Track association for database update
                        result['associations'].append({
                            'model_id': model['id'],
                            'source_path': assoc_file,
                            'target_path': assoc_final_path,
                            'file_name': assoc_filename
                        })
            
            result['success'] = True
            
        except Exception as e:
            print(f"    ❌ Error in _move_model_with_tracking: {e}")
        
        return result
    
    def _apply_batch_database_updates(self, model_updates: list, associations: list) -> None:
        """Apply batch database updates for moved models and associations"""
        if not model_updates and not associations:
            return
        
        try:
            import sqlite3
            import os
            
            # Connect to the configured database
            server_db_path = "Database/civitai.sqlite"
            if not os.path.exists(server_db_path):
                print(f"⚠️  Database not found: {server_db_path}")
                return
            
            conn = sqlite3.connect(server_db_path, timeout=30.0)
            cursor = conn.cursor()
            
            # Process model updates - handle both existing and new models
            if model_updates:
                for update in model_updates:
                    model_id = update['model_id']
                    
                    if model_id == -1:
                        # This is a new model that doesn't exist in model_files yet
                        # First, get the scanned_file_id from the update info
                        scanned_file_id = update.get('scanned_file_id')
                        if scanned_file_id:
                            # Insert new model_files record
                            cursor.execute('''
                                INSERT INTO model_files 
                                (scanned_file_id, model_name, base_model, model_type, target_path, status, is_duplicate, created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, 0, strftime('%s', 'now'), strftime('%s', 'now'))
                            ''', (scanned_file_id, update['model_name'], update['base_model'], 
                                  update.get('model_type', 'LORA'), update['target_path'], update['status']))
                    else:
                        # Update existing model_files record
                        cursor.execute('''
                            UPDATE model_files 
                            SET target_path = ?, status = ?, model_name = ?, base_model = ?, updated_at = strftime('%s', 'now')
                            WHERE id = ?
                        ''', (update['target_path'], update['status'], update['model_name'], update['base_model'], model_id))
            
            # Batch insert/update associations
            if associations:
                # First, check which files exist in scanned_files database
                for assoc in associations:
                    # Check if associated file exists in database
                    cursor.execute('''
                        SELECT id FROM scanned_files WHERE file_path = ? OR file_path = ?
                    ''', (assoc['source_path'], assoc['target_path']))
                    
                    scanned_file_record = cursor.fetchone()
                    
                    if scanned_file_record:
                        scanned_file_id = scanned_file_record[0]
                        
                        # Insert association record
                        cursor.execute('''
                            INSERT OR REPLACE INTO associated_files 
                            (model_file_id, scanned_file_id, association_type, source_path, target_path, is_moved)
                            VALUES (?, ?, 'related', ?, ?, 1)
                        ''', (assoc['model_id'], scanned_file_id, assoc['source_path'], assoc['target_path']))
                    else:
                        # Add new scanned file record for moved associated file
                        import os
                        file_size = 0
                        try:
                            if os.path.exists(assoc['target_path']):
                                file_size = os.path.getsize(assoc['target_path'])
                        except:
                            pass
                        
                        # Get file extension
                        file_ext = os.path.splitext(assoc['file_name'])[1].lower()
                        
                        cursor.execute('''
                            INSERT INTO scanned_files 
                            (file_path, file_name, file_size, sha256, file_type, extension, last_modified, scan_date)
                            VALUES (?, ?, ?, '', 'associated', ?, strftime('%s', 'now'), strftime('%s', 'now'))
                        ''', (assoc['target_path'], assoc['file_name'], file_size, file_ext))
                        
                        new_scanned_file_id = cursor.lastrowid
                        
                        # Insert association record
                        cursor.execute('''
                            INSERT OR REPLACE INTO associated_files 
                            (model_file_id, scanned_file_id, association_type, source_path, target_path, is_moved)
                            VALUES (?, ?, 'related', ?, ?, 1)
                        ''', (assoc['model_id'], new_scanned_file_id, assoc['source_path'], assoc['target_path']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error in batch database update: {e}")
            raise
    
    def _apply_batch_updates(self, db_path: str, updates: list) -> None:
        """Apply a batch of updates to the database"""
        if not updates:
            return
            
        try:
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=30.0)
            cursor = conn.cursor()
            
            # Prepare batch update
            update_data = [(update['model_name'], update['base_model'], update['id']) for update in updates]
            
            cursor.executemany('''
                UPDATE model_files 
                SET model_name = ?, base_model = ?
                WHERE id = ?
            ''', update_data)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error applying batch updates: {e}")
            raise

    def _find_file_comprehensive(self, original_path: str, file_name: str, destination_directory: str | None = None) -> str | None:
        """Comprehensively search for a file using multiple strategies"""
        
        # Strategy 0: Check model sorter database for relocated files
        print(f"  🔍 Strategy 0: Model sorter database lookup")
        try:
            model_sorter_db_path = os.path.join(os.path.dirname(__file__), 'model_sorter.sqlite')
            if os.path.exists(model_sorter_db_path):
                import sqlite3
                model_conn = sqlite3.connect(model_sorter_db_path)
                model_cursor = model_conn.cursor()
                
                # Search for the file in model_files table by filename
                model_cursor.execute('''
                    SELECT target_path, source_path 
                    FROM model_files 
                    WHERE source_path LIKE ? OR target_path LIKE ?
                    ORDER BY target_path IS NOT NULL DESC
                    LIMIT 5
                ''', (f"%{file_name}", f"%{file_name}"))
                
                model_results = model_cursor.fetchall()
                
                if model_results:
                    print(f"    Found {len(model_results)} entries in model sorter database:")
                    for target_path, source_path in model_results:
                        print(f"      Source: {source_path}")
                        print(f"      Target: {target_path}")
                        
                        # Check if target exists (preferred)
                        if target_path and os.path.exists(target_path):
                            print(f"    ✅ Found at target location: {target_path}")
                            model_conn.close()
                            return target_path
                        
                        # Check if source still exists
                        if source_path and os.path.exists(source_path):
                            print(f"    ✅ Found at source location: {source_path}")
                            model_conn.close()
                            return source_path
                
                # Also check associated_files table for metadata files
                model_cursor.execute('''
                    SELECT source_path, target_path
                    FROM associated_files 
                    WHERE source_path LIKE ? OR target_path LIKE ?
                    ORDER BY target_path IS NOT NULL DESC
                    LIMIT 5
                ''', (f"%{file_name}", f"%{file_name}"))
                
                assoc_results = model_cursor.fetchall()
                
                if assoc_results:
                    print(f"    Found {len(assoc_results)} entries in associated files:")
                    for source_path, target_path in assoc_results:
                        print(f"      Source: {source_path}")
                        print(f"      Target: {target_path}")
                        
                        # Check if target exists (preferred)
                        if target_path and os.path.exists(target_path):
                            print(f"    ✅ Found at target location: {target_path}")
                            model_conn.close()
                            return target_path
                        
                        # Check if source still exists
                        if source_path and os.path.exists(source_path):
                            print(f"    ✅ Found at source location: {source_path}")
                            model_conn.close()
                            return source_path
                
                model_conn.close()
                print(f"    ❌ Not found in model sorter database")
            else:
                print(f"    ⚠️  Model sorter database not found at: {model_sorter_db_path}")
        except Exception as e:
            print(f"    ⚠️  Error checking model sorter database: {e}")
        
        # Strategy 0.5: Check destination directory from config.ini
        if destination_directory and os.path.exists(destination_directory):
            print(f"  🔍 Strategy 0.5: Search in config destination directory")
            print(f"    Searching in: {destination_directory}")
            
            try:
                files_checked = 0
                found_matches = []
                
                for root, dirs, files in os.walk(destination_directory):
                    if file_name in files:
                        found_path = os.path.join(root, file_name)
                        found_matches.append(found_path)
                        files_checked += 1
                        
                        # Limit to prevent too many results
                        if files_checked >= 5:
                            break
                
                if found_matches:
                    print(f"    Found {len(found_matches)} potential matches in destination:")
                    for match_path in found_matches:
                        print(f"      Checking: {match_path}")
                        if os.path.exists(match_path):
                            print(f"    ✅ Found in destination: {match_path}")
                            return match_path
                else:
                    print(f"    ❌ Not found in destination directory")
            except Exception as e:
                print(f"    ⚠️  Error searching destination directory: {e}")
        
        # Strategy 1: Try mount point alternatives
        alternative_paths = []
        
        # Convert between /mnt/user/ and /mnt/ patterns
        if '/mnt/user/' in original_path:
            alt_path = original_path.replace('/mnt/user/', '/mnt/')
            alternative_paths.append(alt_path)
        elif '/mnt/' in original_path and '/mnt/user/' not in original_path:
            alt_path = original_path.replace('/mnt/', '/mnt/user/')
            alternative_paths.append(alt_path)
        
        print(f"  🔍 Strategy 1: Mount point alternatives")
        for alt_path in alternative_paths:
            print(f"    Trying: {alt_path}")
            if os.path.exists(alt_path):
                print(f"    ✅ Found at: {alt_path}")
                return alt_path
            print(f"    ❌ Not found")
        
        # Strategy 2: Check if file moved to target/sorted location
        # Look for model_files entries to see if there's a target_path
        try:
            cursor = self.db.conn.cursor()
            
            # Find potential target paths from model_files where this might be an associated file
            base_dir = os.path.dirname(original_path)
            cursor.execute('''
                SELECT DISTINCT mf.target_path 
                FROM model_files mf 
                WHERE mf.source_path LIKE ?
                AND mf.target_path IS NOT NULL
                LIMIT 10
            ''', (f"{base_dir}%",))
            
            target_dirs = cursor.fetchall()
            
            if target_dirs:
                print(f"  🔍 Strategy 2: Check target/sorted locations")
                for (target_dir,) in target_dirs:
                    if target_dir:
                        # Try the file in the target directory
                        target_file_path = os.path.join(os.path.dirname(target_dir), file_name)
                        print(f"    Trying target: {target_file_path}")
                        if os.path.exists(target_file_path):
                            print(f"    ✅ Found at target: {target_file_path}")
                            return target_file_path
                        print(f"    ❌ Not found")
        except Exception as e:
            print(f"    ⚠️  Error checking target locations: {e}")
        
        # Strategy 3: Search by filename in nearby directories
        print(f"  🔍 Strategy 3: Search nearby directories")
        try:
            # Get the parent directories to search
            search_dirs = []
            parts = original_path.split('/')
            
            # Try parent directories at different levels
            for i in range(len(parts) - 1, max(0, len(parts) - 4), -1):
                parent_dir = '/'.join(parts[:i])
                if parent_dir and os.path.exists(parent_dir):
                    search_dirs.append(parent_dir)
            
            for search_dir in search_dirs[:3]:  # Limit to 3 directories to avoid too much searching
                print(f"    Searching in: {search_dir}")
                files_checked = 0
                dirs_checked = 0
                
                for root, dirs, files in os.walk(search_dir):
                    dirs_checked += 1
                    print(f"      📁 Directory [{dirs_checked}]: {root}")
                    print(f"         Files in this directory: {len(files)}")
                    
                    # Show first few files for context
                    if files:
                        sample_files = files[:5]
                        for f in sample_files:
                            files_checked += 1
                            print(f"         - {f}")
                            if f == file_name:
                                found_path = os.path.join(root, file_name)
                                print(f"    ✅ FOUND TARGET FILE: {found_path}")
                                return found_path
                        
                        if len(files) > 5:
                            print(f"         ... and {len(files) - 5} more files")
                            # Check remaining files without printing
                            for f in files[5:]:
                                files_checked += 1
                                if f == file_name:
                                    found_path = os.path.join(root, file_name)
                                    print(f"    ✅ FOUND TARGET FILE: {found_path}")
                                    return found_path
                    else:
                        print(f"         (empty directory)")
                    
                    # Limit depth to avoid infinite searching
                    if root.count('/') - search_dir.count('/') > 3:
                        print(f"         🛑 Max depth reached, stopping deeper search")
                        dirs.clear()  # Don't go deeper
                    
                    # Limit total files checked to avoid excessive output
                    if files_checked > 100:
                        print(f"         🛑 Checked {files_checked} files, limiting search to avoid spam")
                        break
                
                print(f"    📊 Search summary for {search_dir}:")
                print(f"       - Directories checked: {dirs_checked}")
                print(f"       - Files checked: {files_checked}")
                print(f"    ❌ Target file '{file_name}' not found in {search_dir}")
        except Exception as e:
            print(f"    ⚠️  Error searching directories: {e}")
        
        print(f"  ❌ File not found using any strategy")
        return None

    def _process_error_files(self, error_files: list, cursor, results: dict, destination_directory: str | None = None):
        """Process files from previous error log first"""
        print(f"Processing {len(error_files)} files from previous error log...")
        
        fixed_count = 0
        still_missing = []
        
        for error_detail in error_files:
            file_name = error_detail.get('file_name', 'Unknown')
            table = error_detail.get('table', 'scanned_files')
            
            print(f"\n🔄 Retry: {file_name} (from {table})")
            
            if table == 'scanned_files':
                file_id = error_detail.get('file_id')
                db_path = error_detail.get('database_path')
                
                print(f"  📊 Debug info:")
                print(f"     File ID: {file_id}")
                print(f"     Current DB path: {db_path}")
                
                if os.path.exists(db_path):
                    print(f"  ✅ Now exists at original path!")
                    fixed_count += 1
                    continue
                
                found_path = self._find_file_comprehensive(db_path, file_name, destination_directory)
                if found_path:
                    # Check if target path already exists in database (duplicate entry)
                    cursor.execute('SELECT id, file_path FROM scanned_files WHERE file_path = ?', (found_path,))
                    existing_record = cursor.fetchone()
                    
                    if existing_record and existing_record[0] != file_id:
                        print(f"  🔄 DUPLICATE DETECTED:")
                        print(f"     CURRENT RECORD (ID {file_id}): {db_path}")
                        print(f"     EXISTING RECORD (ID {existing_record[0]}): {found_path}")
                        print(f"  🗑️  REMOVING DUPLICATE RECORD (keeping the one with correct path)")
                        
                        # Remove the current record (incorrect path) since the correct one already exists
                        cursor.execute('DELETE FROM scanned_files WHERE id = ?', (file_id,))
                        results['paths_corrected'] += 1
                        results['database_updates'] += 1
                        fixed_count += 1
                        print(f"  ✅ Duplicate record removed successfully")
                    else:
                        # Check if the found path is actually the same as current db_path
                        if found_path == db_path:
                            print(f"  ✅ File already at correct path: {found_path}")
                            fixed_count += 1
                        else:
                            # Update the database with the correct path
                            print(f"  🔄 UPDATING PATH:")
                            print(f"     FROM: {db_path}")
                            print(f"     TO:   {found_path}")
                            
                            try:
                                cursor.execute('''
                                    UPDATE scanned_files 
                                    SET file_path = ?, updated_at = strftime('%s', 'now')
                                    WHERE id = ?
                                ''', (found_path, file_id))
                                
                                results['paths_corrected'] += 1
                                results['database_updates'] += 1
                                fixed_count += 1
                                print(f"  ✅ Fixed: {db_path} → {found_path}")
                            except Exception as e:
                                print(f"  ❌ Error updating path: {e}")
                                still_missing.append(error_detail)
                else:
                    still_missing.append(error_detail)
                    print(f"  ❌ Still missing")
            
            elif table == 'associated_files':
                assoc_id = error_detail.get('assoc_id')
                source_path = error_detail.get('source_path')
                target_path = error_detail.get('target_path')
                
                found_source = None
                found_target = None
                
                if source_path and not os.path.exists(source_path):
                    found_source = self._find_file_comprehensive(source_path, file_name, destination_directory)
                
                if target_path and not os.path.exists(target_path):
                    found_target = self._find_file_comprehensive(target_path, file_name, destination_directory)
                
                if found_source or found_target:
                    new_source = found_source if found_source else source_path
                    new_target = found_target if found_target else target_path
                    
                    cursor.execute('''
                        UPDATE associated_files 
                        SET source_path = ?, target_path = ?
                        WHERE id = ?
                    ''', (new_source, new_target, assoc_id))
                    
                    results['paths_corrected'] += 1
                    results['database_updates'] += 1
                    fixed_count += 1
                    print(f"  ✅ Fixed associated file paths")
                else:
                    still_missing.append(error_detail)
                    print(f"  ❌ Still missing")
        
        print(f"\n📊 Previous errors processed:")
        print(f"  ✅ Fixed: {fixed_count}")
        print(f"  ❌ Still missing: {len(still_missing)}")
        
        # Update error details with still missing files
        results['error_details'].extend(still_missing)
        results['errors'] += len(still_missing)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from INI file"""
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Try DEFAULT section first, then specific sections for backwards compatibility
        def get_config_value(section, key, fallback, value_type='str'):
            try:
                # Try DEFAULT section first - but only if the key actually exists
                if config.has_option('DEFAULT', key):
                    if value_type == 'int':
                        return config.getint('DEFAULT', key)
                    elif value_type == 'bool':
                        return config.getboolean('DEFAULT', key)
                    else:
                        return config.get('DEFAULT', key)
            except:
                pass
            
            # Fallback to specific section
            try:
                if value_type == 'int':
                    return config.getint(section, key, fallback=fallback)
                elif value_type == 'bool':
                    return config.getboolean(section, key, fallback=fallback)
                else:
                    return config.get(section, key, fallback=fallback)
            except:
                return fallback
        
        # Get destination directory and provide alias for backwards compatibility
        dest_dir = get_config_value('Paths', 'destination_directory', './models')
        
        return {
            'source_directory': get_config_value('Paths', 'source_directory', 'example-models'),
            'destination_directory': dest_dir,
            'target_directory': dest_dir,  # Alias for backwards compatibility
            'database_path': get_config_value('Paths', 'database_path', 'Database/file_scanner.sqlite'),
            'model_extensions': self.MODEL_EXTENSIONS,
            'related_extensions': set(ext.strip() for ext in 
                                     str(get_config_value('Processing', 'related_extensions',
                                               '.preview.png,.webp,.jpg,.jpeg,.png,.civitai.info,.metadata.json,.txt,.yaml,.yml,.json')).split(',')),
            'hash_cache_file': get_config_value('Paths', 'hash_cache_file', 'model_hashes.json'),
            'hash_cache_save_interval': get_config_value('Processing', 'hash_cache_save_interval', 10, 'int'),
            'max_file_size': get_config_value('Processing', 'max_file_size', 0, 'int'),
            'verbose': get_config_value('Logging', 'verbose', True, 'bool'),
            'show_progress': get_config_value('Logging', 'show_progress', True, 'bool'),
            'database_commit_batch_size': get_config_value('Logging', 'database_commit_batch_size', 1000, 'int'),
            'directory_batch_size': get_config_value('Logging', 'directory_batch_size', 100, 'int'),
            'model_sorting_batch_size': get_config_value('Logging', 'model_sorting_batch_size', 100, 'int'),
            'use_model_type_subfolders': get_config_value('Processing', 'use_model_type_subfolders', True, 'bool')
        }
    
    def calculate_hashes(self, file_path: str, force_blake3: bool = False) -> Tuple[str, Optional[str], Optional[str]]:
        """Calculate SHA256, AutoV3, and BLAKE3 hashes for a file
        
        Args:
            file_path: Path to the file to hash
            force_blake3: If True, always calculate BLAKE3 even for SafeTensors
        
        Returns:
            Tuple[str, Optional[str], Optional[str]]: (sha256_hash, autov3_hash, blake3_hash)
            - AutoV3 hash is only for SafeTensors files
            - BLAKE3 hash is for non-SafeTensors or when force_blake3=True
        """
        # Check cache first
        file_stat = os.stat(file_path)
        cache_key = f"{file_path}:{file_stat.st_size}:{file_stat.st_mtime}"
        
        if cache_key in self.hash_cache:
            cached_value = self.hash_cache[cache_key]
            # Handle old cache format (just SHA256) vs new format (tuple)
            if isinstance(cached_value, str):
                sha256_hash = cached_value
                # Compute AutoV3 for SafeTensors files if not already cached
                autov3_hash = None
                if file_path.lower().endswith('.safetensors'):
                    autov3_hash = compute_autov3_hex(Path(file_path))
                    # Update cache with new format
                    self.hash_cache[cache_key] = (sha256_hash, autov3_hash)
                return sha256_hash, autov3_hash, None
            else:
                # cached_value is a tuple, extend it to 3 elements if needed
                if len(cached_value) == 2:
                    return cached_value[0], cached_value[1], None
                return cached_value
        
        hash_sha256 = hashlib.sha256()
        autov3_hash = None
        blake3_hash = None
        is_safetensors = file_path.lower().endswith('.safetensors')
        
        try:
            with open(file_path, 'rb') as f:
                if is_safetensors and not force_blake3:
                    # OPTIMIZED: Single-pass reading for SafeTensors files (SHA256 + AutoV3)
                    # Read header to determine AutoV3 offset
                    header8 = f.read(8)
                    if len(header8) == 8:
                        try:
                            header_size = int.from_bytes(header8, "little")
                            offset = header_size + 8
                            
                            # Check if valid SafeTensors
                            f.seek(0, 2)
                            filesize = f.tell()
                            if offset < filesize:
                                # Setup AutoV3 hasher
                                autov3_hasher = hashlib.sha256()
                                
                                # Reset to beginning and process entire file in single pass
                                f.seek(0)
                                bytes_read = 0
                                
                                # Use 1MB chunks for optimal performance (based on benchmark)
                                for chunk in iter(lambda: f.read(1048576), b""):
                                    # Always update SHA256 with full file
                                    hash_sha256.update(chunk)
                                    
                                    # For AutoV3, only update with bytes after offset
                                    if bytes_read >= offset:
                                        # Entire chunk is after offset
                                        autov3_hasher.update(chunk)
                                    elif bytes_read + len(chunk) > offset:
                                        # Chunk spans the offset - use partial chunk
                                        chunk_start = offset - bytes_read
                                        autov3_hasher.update(chunk[chunk_start:])
                                    
                                    bytes_read += len(chunk)
                                
                                autov3_hash = autov3_hasher.hexdigest().upper()
                            else:
                                # Invalid SafeTensors - fall back to SHA256 only
                                f.seek(0)
                                for chunk in iter(lambda: f.read(1048576), b""):
                                    hash_sha256.update(chunk)
                        except Exception:
                            # Error reading SafeTensors header - fall back to SHA256 only
                            f.seek(0)
                            for chunk in iter(lambda: f.read(1048576), b""):
                                hash_sha256.update(chunk)
                    else:
                        # Invalid header - fall back to SHA256 only
                        f.seek(0)
                        for chunk in iter(lambda: f.read(1048576), b""):
                            hash_sha256.update(chunk)
                else:
                    # Non-SafeTensors files OR force_blake3: Use BLAKE3 + SHA256
                    blake3_hasher = None
                    if BLAKE3_AVAILABLE and blake3 is not None:
                        blake3_hasher = blake3.blake3()
                    
                    # Single pass for both SHA256 and BLAKE3
                    for chunk in iter(lambda: f.read(1048576), b""):
                        hash_sha256.update(chunk)
                        if blake3_hasher:
                            blake3_hasher.update(chunk)
                    
                    if blake3_hasher:
                        blake3_hash = blake3_hasher.hexdigest().upper()
            
            sha256_hex = hash_sha256.hexdigest().upper()
            
            # Cache all three hashes
            result = (sha256_hex, autov3_hash, blake3_hash)
            self.hash_cache[cache_key] = result
            
            return result
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            self.db.log_operation('hash_calculation', file_path, 'error', str(e))
            return ("", None, None)
    
    def calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA256 hash for a file (backward compatibility)"""
        sha256_hash, _, _ = self.calculate_hashes(file_path)
        return sha256_hash
    
    def save_hash_cache(self):
        """Save hash cache to file"""
        try:
            with open(self.config['hash_cache_file'], 'w') as f:
                json.dump(self.hash_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save hash cache: {e}")
    
    def classify_file_type(self, file_path: str, extension: str) -> str:
        """Classify file type based on extension"""
        ext_lower = extension.lower()
        
        if ext_lower in self.MODEL_EXTENSIONS:
            return 'model'
        elif ext_lower in self.IMAGE_EXTENSIONS:
            return 'image'
        elif ext_lower in self.TEXT_EXTENSIONS or '.info' in ext_lower:
            return 'text'
        else:
            return 'other'
    
    def find_associated_files(self, model_path: str) -> List[str]:
        """Find files associated with a model file using multiple correlation methods"""
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        associated_files = []
        
        # METHOD 1: Direct base name matching (existing logic)
        for ext in self.ASSOCIATED_EXTENSIONS:
            candidate_path = os.path.join(model_dir, f"{model_name}{ext}")
            if os.path.exists(candidate_path):
                associated_files.append(candidate_path)
        
        # METHOD 2: Check if this is a multi-model directory requiring advanced correlation
        model_files_in_dir = []
        other_files_in_dir = []
        
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isfile(item_path):
                _, ext = os.path.splitext(item)
                if ext.lower() in self.MODEL_EXTENSIONS:
                    model_files_in_dir.append(item_path)
                elif ext.lower() in self.ASSOCIATED_EXTENSIONS:
                    other_files_in_dir.append(item_path)
        
        # Use advanced correlation if multiple models OR if there are unmatched files in single-model directory
        unmatched_files = [f for f in other_files_in_dir 
                         if os.path.splitext(os.path.basename(f))[0] != model_name]
        
        if len(model_files_in_dir) > 1 or (len(model_files_in_dir) == 1 and len(unmatched_files) > 0):
            if self.config['verbose']:
                if len(model_files_in_dir) > 1:
                    print(f"Multi-model directory detected: {len(model_files_in_dir)} models, {len(other_files_in_dir)} media/metadata files")
                else:
                    print(f"Single-model directory with unmatched files: 1 model, {len(unmatched_files)} unmatched files")
            
            # METHOD 3: Correlate via metadata analysis for unmatched files
            
            if unmatched_files:
                correlated_files = self._correlate_files_with_model(model_path, unmatched_files)
                associated_files.extend(correlated_files)
        
        # METHOD 4: Look for files in subdirectories with related names (existing logic)
        if os.path.isdir(model_dir):
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    # Check if directory name relates to model
                    if any(pattern in item.lower() for pattern in 
                          ['image', 'file', 'data', 'doc', 'sample']) or model_name.lower() in item.lower():
                        # Scan subdirectory for associated files
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isfile(subitem_path):
                                _, ext = os.path.splitext(subitem)
                                if ext.lower() in self.ASSOCIATED_EXTENSIONS:
                                    associated_files.append(subitem_path)
        
        return list(set(associated_files))  # Remove duplicates
    
    def _correlate_files_with_model(self, model_path: str, candidate_files: List[str]) -> List[str]:
        """Correlate media/metadata files with a model using advanced analysis"""
        correlated_files = []
        
        try:
            # Calculate model hashes for correlation
            model_sha256, model_autov3, _ = self.calculate_hashes(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            if self.config['verbose']:
                print(f"Correlating {len(candidate_files)} files with model: {model_name}")
            
            for file_path in candidate_files:
                try:
                    correlation_score = 0
                    correlation_reasons = []
                    
                    # METHOD 1: Extract and analyze image metadata for model references
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_metadata = extract_image_metadata(file_path)
                        if image_metadata and image_metadata.get('ai_metadata'):
                            comprehensive_metadata = extract_comprehensive_metadata(file_path, image_metadata)
                            
                            # Check for model hash matches
                            if comprehensive_metadata.get('model_hash'):
                                model_hash_meta = comprehensive_metadata['model_hash'].upper()
                                if (model_hash_meta == model_sha256.upper() or 
                                    (model_autov3 and model_hash_meta == model_autov3.upper())):
                                    correlation_score += 100
                                    correlation_reasons.append("model_hash_match")
                            
                            # Check for model name matches
                            if comprehensive_metadata.get('model_name'):
                                model_name_meta = comprehensive_metadata['model_name'].lower()
                                if model_name.lower() in model_name_meta or model_name_meta in model_name.lower():
                                    correlation_score += 50
                                    correlation_reasons.append("model_name_similarity")
                    
                    # METHOD 2: Folder-based correlation (single model in named folder)
                    model_dir = os.path.dirname(model_path)
                    folder_name = os.path.basename(model_dir).lower()
                    model_name_clean = model_name.lower().replace(' ', '-').replace('_', '-')
                    
                    # If folder name relates to model name and it's the only model in folder
                    if (folder_name in model_name_clean or model_name_clean in folder_name or 
                        any(word in folder_name for word in model_name_clean.split('-') if len(word) > 3)):
                        correlation_score += 60
                        correlation_reasons.append("folder_name_correlation")
                    
                    # METHOD 3: Check filename patterns for Civitai IDs
                    filename = os.path.basename(file_path)
                    if filename.isdigit() or any(char.isdigit() for char in filename[:8]):
                        # Potential Civitai ID - try cross-reference with civitai database
                        civitai_correlation = self._check_civitai_correlation(file_path, model_sha256, model_autov3 or "")
                        if civitai_correlation:
                            correlation_score += 75
                            correlation_reasons.append("civitai_correlation")
                    
                    # METHOD 4: Enhanced metadata file analysis
                    if file_path.lower().endswith(('.json', '.txt', '.metadata.json')):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            # Parse JSON for structured metadata analysis
                            metadata_content = {}
                            try:
                                if file_path.lower().endswith('.json'):
                                    metadata_content = json.loads(content)
                            except:
                                pass
                            
                            # Look for hash references in metadata
                            hash_found = model_sha256.lower() in content.lower()
                            if model_autov3:
                                hash_found = hash_found or model_autov3.lower() in content.lower()
                            if hash_found:
                                correlation_score += 90
                                correlation_reasons.append("metadata_hash_reference")
                            
                            # Advanced model name correlation
                            elif model_name.lower() in content.lower():
                                correlation_score += 30
                                correlation_reasons.append("metadata_name_reference")
                            
                            # Enhanced metadata field analysis for .metadata.json files
                            if isinstance(metadata_content, dict):
                                # Check for model-specific metadata fields
                                model_identifier_fields = [
                                    'model_name', 'name', 'title', 'filename', 'base_model',
                                    'sha256', 'model_hash', 'hash', 'model_id', 'version_id'
                                ]
                                
                                for field in model_identifier_fields:
                                    if field in metadata_content:
                                        field_value = str(metadata_content[field]).lower()
                                        # Check for exact model name match
                                        if model_name.lower() == field_value:
                                            correlation_score += 80
                                            correlation_reasons.append(f"metadata_exact_{field}_match")
                                        # Check for partial model name match
                                        elif model_name.lower() in field_value or field_value in model_name.lower():
                                            correlation_score += 40
                                            correlation_reasons.append(f"metadata_partial_{field}_match")
                                
                                # Check for hash matches in structured fields
                                hash_fields = ['sha256', 'model_hash', 'hash', 'hashes']
                                for field in hash_fields:
                                    if field in metadata_content:
                                        field_value = str(metadata_content[field]).lower()
                                        if model_sha256.lower() in field_value or (model_autov3 and model_autov3.lower() in field_value):
                                            correlation_score += 95
                                            correlation_reasons.append(f"metadata_structured_hash_match_{field}")
                                
                                # Check description and content fields for character/theme correlation
                                content_fields = ['description', 'notes', 'tags', 'trigger_words', 'trained_words']
                                model_base_words = set(word.lower() for word in model_name.replace('_', ' ').replace('-', ' ').split() if len(word) > 2)
                                
                                for field in content_fields:
                                    if field in metadata_content:
                                        field_text = str(metadata_content[field]).lower()
                                        # Count matching significant words
                                        matching_words = sum(1 for word in model_base_words if word in field_text)
                                        if matching_words >= 2:  # At least 2 significant words match
                                            correlation_score += min(matching_words * 15, 50)
                                            correlation_reasons.append(f"metadata_content_correlation_{field}")
                                
                                # Special handling for Disney/character-based models
                                if any(keyword in model_name.lower() for keyword in ['disney', 'princess', 'character', 'cgi']):
                                    character_indicators = ['character', 'princess', 'disney', 'cgi', 'cartoon', 'animation']
                                    description_text = str(metadata_content.get('description', '')).lower()
                                    if any(indicator in description_text for indicator in character_indicators):
                                        correlation_score += 25
                                        correlation_reasons.append("character_theme_correlation")
                                
                        except Exception as e:
                            if self.config['verbose']:
                                print(f"    Warning: Error analyzing metadata {file_path}: {e}")
                            pass
                    
                    # If correlation score is high enough, include the file
                    if correlation_score >= 50:
                        correlated_files.append(file_path)
                        if self.config['verbose']:
                            print(f"  Correlated {os.path.basename(file_path)} (score: {correlation_score}, reasons: {correlation_reasons})")
                
                except Exception as e:
                    if self.config['verbose']:
                        print(f"  Error correlating {file_path}: {e}")
        
        except Exception as e:
            if self.config['verbose']:
                print(f"Error in file correlation: {e}")
        
        return correlated_files
    
    def _check_civitai_correlation(self, file_path: str, model_sha256: str, model_autov3: str) -> bool:
        """Check if file correlates with model via Civitai database cross-reference"""
        try:
            # Extract potential Civitai ID from filename
            filename = os.path.basename(file_path)
            civitai_id_match = re.search(r'(\d{6,})', filename)
            
            if civitai_id_match:
                civitai_id = civitai_id_match.group(1)
                
                # TODO: Cross-reference with civitai.sqlite database
                # This would require connecting to the civitai database and checking
                # if the model hash exists in the same model/version as the image ID
                # For now, return basic numeric ID correlation
                return len(civitai_id) >= 6
            
            return False
        except Exception:
            return False
    
    def link_file_associations(self, scanned_file_id: int, file_path: str, file_type: str) -> None:
        """Link a newly scanned file with potential model associations"""
        if file_type == 'model':
            # If this is a model file, find associated files for it
            self._link_model_associations(scanned_file_id, file_path)
        else:
            # If this is a potential associated file, find models it could belong to
            self._link_associated_file_to_models(scanned_file_id, file_path)
    
    def _link_model_associations(self, model_scanned_file_id: int, model_path: str) -> None:
        """Find and link associated files for a model"""
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)
        model_base_name = os.path.splitext(model_filename)[0]
        
        cursor = self.db.conn.cursor()
        
        # Find potential associated files in the same directory
        cursor.execute('''
            SELECT id, file_path, file_name
            FROM scanned_files
            WHERE file_path LIKE ? AND file_type IN ('text', 'image', 'video', 'json', 'unknown')
            AND id != ?
        ''', (f"{model_dir}%", model_scanned_file_id))
        
        potential_files = cursor.fetchall()
        
        for assoc_id, assoc_path, assoc_name in potential_files:
            # Check if in same directory
            if os.path.dirname(assoc_path) != model_dir:
                continue
                
            assoc_base_name = os.path.splitext(assoc_name)[0]
            
            # Exact name match
            if assoc_base_name == model_base_name:
                assoc_type = self._determine_association_type(assoc_path)
                self._create_association(model_scanned_file_id, assoc_id, assoc_type, assoc_path)
    
    def _link_associated_file_to_models(self, assoc_scanned_file_id: int, assoc_path: str) -> None:
        """Find and link models that this associated file could belong to"""
        assoc_dir = os.path.dirname(assoc_path)
        assoc_filename = os.path.basename(assoc_path)
        assoc_base_name = os.path.splitext(assoc_filename)[0]
        
        cursor = self.db.conn.cursor()
        
        # Find model files in the same directory
        cursor.execute('''
            SELECT id, file_path, file_name
            FROM scanned_files
            WHERE file_path LIKE ? AND file_type = 'model'
            AND id != ?
        ''', (f"{assoc_dir}%", assoc_scanned_file_id))
        
        potential_models = cursor.fetchall()
        models_in_dir = [m for m in potential_models if os.path.dirname(m[1]) == assoc_dir]
        
        if len(models_in_dir) == 1:
            # Only one model in directory - link to it
            model_id, model_path, model_name = models_in_dir[0]
            assoc_type = self._determine_association_type(assoc_path)
            self._create_association(model_id, assoc_scanned_file_id, assoc_type, assoc_path)
        else:
            # Multiple models - only link if exact name match
            for model_id, model_path, model_name in models_in_dir:
                model_base_name = os.path.splitext(model_name)[0]
                if model_base_name == assoc_base_name:
                    assoc_type = self._determine_association_type(assoc_path)
                    self._create_association(model_id, assoc_scanned_file_id, assoc_type, assoc_path)
                    break
    
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
    
    def _create_association(self, model_scanned_file_id: int, assoc_scanned_file_id: int, 
                          assoc_type: str, source_path: str) -> None:
        """Create an association record in the temp table (will be linked to model_files later)"""
        cursor = self.db.conn.cursor()
        
        # Create temporary table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS temp_file_associations (
                model_scanned_file_id INTEGER,
                assoc_scanned_file_id INTEGER,
                association_type TEXT,
                source_path TEXT,
                created_at INTEGER,
                PRIMARY KEY (model_scanned_file_id, assoc_scanned_file_id)
            )
        ''')
        
        # Insert association
        cursor.execute('''
            INSERT OR REPLACE INTO temp_file_associations 
            (model_scanned_file_id, assoc_scanned_file_id, association_type, source_path, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_scanned_file_id, assoc_scanned_file_id, assoc_type, source_path, int(time.time())))
        
        # Don't commit here - let the scanner handle batched commits
    
    def scan_directory_hierarchical(self, root_directory: str, fast_prescan: bool = False, force_blake3: bool = False, folder_limit: Optional[int] = None, verbose_dry_run: bool = False) -> Dict[str, List[Dict]]:
        """
        Hierarchical directory scanning: scans from deepest subfolders first,
        marks completed directories, and propagates completion status upward.
        
        Args:
            root_directory: Root directory to scan hierarchically
            fast_prescan: If True, only calculate AutoV3 hashes for fast pre-scanning
            force_blake3: If True, force BLAKE3 rescanning for non-SafeTensors files
            folder_limit: If specified, only process this many folders per run for incremental scanning
            verbose_dry_run: If True, show every file check and folder action (for dry-run mode)
        """
        if not os.path.exists(root_directory):
            print(f"Directory not found: {root_directory}")
            return {'models': [], 'images': [], 'text_files': [], 'other_files': []}
        
        # Check backward compatibility - do we have directory tracking tables?
        has_directory_tables = self._check_directory_tables_exist()
        
        # Get all directories in hierarchical order (deepest first)
        # Note: We don't skip the root directory entirely - individual subdirectories
        # will be checked and skipped if they're already complete
        all_directories = self._get_directories_depth_first(root_directory, verbose_dry_run)
        
        # Apply folder limit for incremental scanning
        if folder_limit and folder_limit > 0:
            # Find directories that haven't been processed yet
            if has_directory_tables:
                unprocessed_dirs = []
                if verbose_dry_run and (self.verbose or self.config.get('verbose', True)):
                    print(f"🔍 Filtering directories for folder limit ({folder_limit})...")
                
                for dir_path in all_directories:
                    is_complete, reason = self.db.check_directory_for_changes(dir_path)
                    if not is_complete:
                        unprocessed_dirs.append(dir_path)
                        if verbose_dry_run and (self.verbose or self.config.get('verbose', True)):
                            status_msg = {
                                "new_directory": "📂 New directory",
                                "incomplete": "📂 Incomplete directory", 
                                "additional_files_found": "🔄 Additional files found",
                                "mtime_error": "⚠️ Cannot check modification time"
                            }.get(reason, "📂 Needs processing")
                            print(f"  {status_msg}: {dir_path}")
                    elif verbose_dry_run and (self.verbose or self.config.get('verbose', True)):
                        print(f"  ✅ Directory up-to-date: {dir_path}")
                
                total_unprocessed = len(unprocessed_dirs)
                all_directories = unprocessed_dirs[:folder_limit]
                
                if verbose_dry_run and (self.verbose or self.config.get('verbose', True)):
                    print(f"📊 Folder filtering complete: {total_unprocessed} unprocessed, taking first {len(all_directories)}")
            else:
                # Fallback: just limit total directories
                if verbose_dry_run and (self.verbose or self.config.get('verbose', True)):
                    print(f"🔍 Legacy mode: limiting to first {folder_limit} directories")
                all_directories = all_directories[:folder_limit]
            
            if self.config.get('verbose', True) and folder_limit:
                print(f"� Folder limit applied: Processing {len(all_directories)} of {folder_limit} requested folders")
        
        if self.config.get('verbose', True):
            total_dirs_msg = f"{len(all_directories)} directories"
            if not has_directory_tables:
                total_dirs_msg += " (legacy mode - no directory tracking)"
            print(f"📁 Hierarchical scan starting: {total_dirs_msg} found")
            print(f"🔍 Scanning from deepest subfolders to root: {root_directory}")
        
        total_results = {'models': [], 'images': [], 'text_files': [], 'other_files': []}
        directories_completed = 0
        
        # Process directories from deepest to shallowest
        for directory_path in all_directories:
            # Check directory status with detailed reasons
            if self.skip_folders and not self.force_rescan and not force_blake3 and not verbose_dry_run:
                if has_directory_tables:
                    is_complete, reason = self.db.check_directory_for_changes(directory_path)
                    if is_complete and reason == "up_to_date":
                        if self.config.get('verbose', True):
                            print(f"  ✅ Directory up-to-date: {directory_path}")
                        continue
                    elif not is_complete and reason == "additional_files_found":
                        if self.config.get('verbose', True):
                            print(f"  🔄 Additional files found: {directory_path}")
                    elif not is_complete and reason == "new_directory":
                        if self.config.get('verbose', True):
                            print(f"  🆕 New directory: {directory_path}")
                    elif not is_complete and reason == "incomplete":
                        if self.config.get('verbose', True):
                            print(f"  📂 Incomplete directory: {directory_path}")
                elif not has_directory_tables and self._is_directory_complete_legacy(directory_path):
                    if self.config.get('verbose', True):
                        print(f"  ✅ Directory up-to-date (legacy): {directory_path}")
                    continue
            elif verbose_dry_run and self.skip_folders and not self.force_rescan and not force_blake3:
                # In verbose dry-run mode, show what would normally be skipped but process anyway
                if has_directory_tables:
                    is_complete, reason = self.db.check_directory_for_changes(directory_path)
                    if is_complete and reason == "up_to_date":
                        if self.config.get('verbose', True):
                            print(f"  🔍 DRY-RUN: Directory up-to-date, but checking: {directory_path}")
                    elif not is_complete and reason == "additional_files_found":
                        if self.config.get('verbose', True):
                            print(f"  🔍 DRY-RUN: Additional files found, checking: {directory_path}")
                elif not has_directory_tables and self._is_directory_complete_legacy(directory_path):
                    if self.config.get('verbose', True):
                        print(f"  🔍 DRY-RUN: Directory up-to-date (legacy), but checking: {directory_path}")
            
            # Scan this specific directory (non-recursive)
            dir_results = self._scan_single_directory(directory_path, fast_prescan, force_blake3, verbose_dry_run)
            
            # Merge results
            for key in total_results:
                total_results[key].extend(dir_results[key])
            
            # Mark directory as completed (with backward compatibility)
            total_files = sum(len(dir_results[key]) for key in dir_results)
            
            if has_directory_tables:
                self.db.update_directory_scan_status(directory_path, scan_complete=True, 
                                                   total_files=total_files, scanned_files=total_files)
                # Handle batched commits for directory completions
                self.directory_batch_count += 1
                if self.directory_batch_count >= self.directory_batch_size:
                    self.db.commit()
                    self.directory_batch_count = 0
                if self.config.get('verbose', True):
                    print(f"  ✅ Directory completed and marked: {directory_path} ({total_files} files)")
            else:
                # Legacy mode: just report completion (no database tracking)
                if self.config.get('verbose', True):
                    print(f"  ✅ Directory processed: {directory_path} ({total_files} files) [legacy mode]")
            
            directories_completed += 1
            
            # Check if parent directory is now fully complete (only if we have directory tables)
            if has_directory_tables:
                self._check_and_mark_parent_completion(directory_path, folder_limit)
        
        # Final commit for remaining directory batch
        if has_directory_tables and self.directory_batch_count > 0:
            self.db.commit()
            if self.config.get('verbose', True):
                print(f"💾 Final database commit for {self.directory_batch_count} remaining directory completions")
        
        if self.config.get('verbose', True):
            total_files_processed = sum(len(total_results[key]) for key in total_results)
            print(f"🎯 Hierarchical scan complete: {directories_completed} directories, {total_files_processed} files processed")
        
        return total_results

    def _get_directories_depth_first(self, root_directory: str, verbose_dry_run: bool = False) -> List[str]:
        """Get directories that need scanning using smart hierarchical approach"""
        if self.verbose or verbose_dry_run or self.config.get('verbose', True):
            print(f"🗄️  Checking database for already-scanned directories under: {root_directory}")
        
        # Use hierarchical checking to avoid checking every deep folder
        dirs_to_process = self._check_directories_hierarchically(root_directory, verbose_dry_run)
        
        # Sort by depth (deepest first) then alphabetically for processing
        dirs_to_process.sort(key=lambda x: (-x.count(os.sep), x))
        
        if self.verbose or self.config.get('verbose', True):
            print(f"📊 Hierarchical analysis complete: {len(dirs_to_process)} directories need processing")
        
        return dirs_to_process
    
    def _check_directories_hierarchically(self, root_directory: str, verbose_dry_run: bool = False) -> List[str]:
        """Check directories hierarchically - start from top and only go deeper if changes detected"""
        dirs_needing_processing = []
        known_dirs_cache = {}  # Cache database results
        
        # Get all known directories from database for this root
        known_dirs_from_db = self._get_known_directories_from_database(root_directory)
        up_to_date_count = 0
        rescan_count = 0
        new_count = 0
        
        if self.verbose or verbose_dry_run:
            print(f"🗄️  Found {len(known_dirs_from_db)} directories in database")
            print(f"🔍 Starting hierarchical check from top-level...")
            print(f"🔍 Discovering new directories not yet in database...")
        
        # Build cache of known directories for fast lookup
        for dir_path in known_dirs_from_db:
            known_dirs_cache[dir_path] = True
        
        # Start hierarchical traversal from root
        def check_directory_tree(current_dir: str, depth: int = 0) -> bool:
            """Returns True if this directory tree needs processing"""
            nonlocal up_to_date_count, rescan_count, new_count
            
            indent = "  " * depth
            
            # Check if directory exists
            if not os.path.exists(current_dir):
                if current_dir in known_dirs_cache:
                    if self.verbose:
                        print(f"{indent}❌ Directory no longer exists: {current_dir}")
                return False
            
            # Check if this directory is known in database
            if current_dir in known_dirs_cache:
                # Check if this known directory has changes
                is_complete, reason = self.db.check_directory_for_changes(current_dir)
                
                if is_complete and reason == "up_to_date":
                    up_to_date_count += 1
                    if self.verbose or verbose_dry_run:
                        print(f"{indent}✅ Directory up-to-date (skipping subtree): {current_dir}")
                    
                    # Directory is up-to-date, so skip all subdirectories
                    # If any subdirectory had changes, the parent mtime would have changed
                    return False  # Skip checking subdirectories
                else:
                    # Directory has changes - mark for processing
                    rescan_count += 1
                    dirs_needing_processing.append(current_dir)
                    
                    if reason == "additional_files_found":
                        if self.verbose or verbose_dry_run:
                            print(f"{indent}🔄 Additional files found: {current_dir}")
                    else:
                        if self.verbose or verbose_dry_run:
                            print(f"{indent}📂 Directory needs processing ({reason}): {current_dir}")
                    
                    # Continue checking subdirectories since this directory changed
                    return True
            else:
                # New directory not in database
                new_count += 1
                dirs_needing_processing.append(current_dir)
                if self.verbose or verbose_dry_run:
                    print(f"{indent}🆕 New directory found: {current_dir}")
                return True  # Continue checking subdirectories
        
        # Start recursive traversal from root directory
        def traverse_directory_tree(current_dir: str, depth: int = 0):
            """Recursively traverse directory tree checking each level"""
            should_continue = check_directory_tree(current_dir, depth)
            
            # If directory needs processing or we should continue checking subdirectories
            if should_continue:
                try:
                    # Get immediate subdirectories only
                    subdirs = [
                        os.path.join(current_dir, d) 
                        for d in os.listdir(current_dir) 
                        if os.path.isdir(os.path.join(current_dir, d))
                        and not d.startswith('.')  # Skip hidden directories
                        and d not in {'.cache', '__pycache__', 'node_modules', '.git'}  # Skip system dirs
                    ]
                    
                    if self.verbose and subdirs and depth == 0:
                        print(f"    🔍 Checking {len(subdirs)} subdirectories in: {current_dir}")
                    
                    # Recursively check each subdirectory
                    for subdir in subdirs:
                        if self.verbose and depth == 0:
                            print(f"      📁 Checking: {subdir}")
                        traverse_directory_tree(subdir, depth + 1)
                        
                except (OSError, PermissionError) as e:
                    if self.verbose:
                        print(f"  ⚠️  Cannot read directory: {current_dir} ({e})")
        
        # Start traversal
        traverse_directory_tree(root_directory)
        
        if self.verbose or self.config.get('verbose', True):
            print(f"📋 Hierarchical check complete: {rescan_count} need rescanning, {new_count} new, {up_to_date_count} up-to-date")
        
        return dirs_needing_processing
    
    def _get_known_directories_from_database(self, root_directory: str) -> List[str]:
        """Get all directories from database that are under the root directory"""
        cursor = self.db.conn.cursor()
        # Use LIKE with % wildcard to find all paths under root_directory
        root_pattern = root_directory.rstrip('/') + '/%'
        
        cursor.execute('''
            SELECT DISTINCT directory_path FROM directory_scan_status 
            WHERE directory_path = ? OR directory_path LIKE ?
            ORDER BY directory_path
        ''', (root_directory, root_pattern))
        
        return [row[0] for row in cursor.fetchall()]
    
    def _check_directories_for_changes(self, known_directories: List[str], verbose_dry_run: bool = False) -> List[str]:
        """Check which known directories need rescanning due to modification time changes"""
        dirs_needing_rescan = []
        up_to_date_count = 0
        
        if self.verbose or verbose_dry_run:
            print(f"🔍 Checking {len(known_directories)} known directories for changes...")
        
        for directory in known_directories:
            if not os.path.exists(directory):
                # Directory no longer exists, skip it
                if self.verbose:
                    print(f"  ❌ Directory no longer exists: {directory}")
                continue
                
            # Use the existing is_directory_fully_scanned method which checks mtime
            if not self.db.is_directory_fully_scanned(directory):
                dirs_needing_rescan.append(directory)
                if self.verbose or (verbose_dry_run and self.config.get('verbose', True)):
                    print(f"  🔄 Directory needs rescanning: {directory}")
            else:
                up_to_date_count += 1
                if self.verbose or (verbose_dry_run and self.config.get('verbose', True)):
                    print(f"  ✅ Directory up-to-date: {directory}")
        
        if self.verbose or self.config.get('verbose', True):
            print(f"📋 Status check: {len(dirs_needing_rescan)} need rescanning, {up_to_date_count} up-to-date")
        
        return dirs_needing_rescan
    
    def _discover_new_directories(self, root_directory: str, known_directories: List[str], verbose_dry_run: bool = False) -> List[str]:
        """Discover new directories not yet in database using efficient filesystem scan"""
        new_dirs = []
        ignored_count = 0
        known_dirs_set = set(known_directories)
        
        if self.verbose or verbose_dry_run or self.config.get('verbose', True):
            print(f"🔍 Discovering new directories not yet in database...")
        
        # Directories to ignore (cache, temp, system directories)
        ignored_dir_patterns = {
            '.cache', '__pycache__', '.git', '.svn', '.hg', 
            'node_modules', '.tmp', 'temp', 'cache', '.DS_Store',
            '.vscode', '.idea', 'Thumbs.db'
        }
        
        for dirpath, dirnames, filenames in os.walk(root_directory):
            # Filter out ignored directories from further traversal
            dirnames[:] = [d for d in dirnames if d not in ignored_dir_patterns]
            
            # Check if current directory should be ignored
            dir_name = os.path.basename(dirpath)
            should_ignore = False
            
            if dir_name in ignored_dir_patterns:
                should_ignore = True
            elif '/.cache/' in dirpath or dirpath.endswith('/.cache'):
                should_ignore = True
            
            if should_ignore:
                ignored_count += 1
                # Show detailed ignore info in verbose mode
                if self.verbose and ignored_count <= 20:
                    print(f"  🚫 Ignoring: {dirpath}")
                elif ignored_count % 1000 == 0:
                    if self.verbose or self.config.get('verbose', True):
                        print(f"  🚫 Ignored {ignored_count} cache/system directories so far...")
                elif verbose_dry_run and ignored_count <= 10:
                    # Show first 10 in verbose dry-run mode for debugging
                    if self.config.get('verbose', True):
                        print(f"  🚫 Ignoring: {dirpath}")
                continue
            
            # Only add directories that contain files AND are not already known
            if (filenames or not dirnames) and dirpath not in known_dirs_set:
                new_dirs.append(dirpath)
                if self.verbose or verbose_dry_run or self.config.get('verbose', True):
                    file_count = len(filenames) if filenames else 0
                    subdir_count = len(dirnames) if dirnames else 0
                    print(f"  📁 New directory found: {dirpath} ({file_count} files, {subdir_count} subdirs)")
        
        if self.verbose or self.config.get('verbose', True):
            if ignored_count > 0:
                print(f"🆕 New directory discovery: {len(new_dirs)} new directories found, {ignored_count} cache/system directories ignored")
            else:
                print(f"🆕 New directory discovery: {len(new_dirs)} new directories found")
        
        return new_dirs
    
    def _scan_single_directory(self, directory: str, fast_prescan: bool = False, force_blake3: bool = False, verbose_dry_run: bool = False) -> Dict[str, List[Dict]]:
        """Scan a single directory (non-recursive) for files"""
        results = {'models': [], 'images': [], 'text_files': [], 'other_files': []}
        
        try:
            # Get files in this directory only (not subdirectories)
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            
            for file_name in sorted(files):
                file_path = os.path.join(directory, file_name)
                file_size = os.path.getsize(file_path)
                
                # Skip metadata files that should not be processed as regular files
                metadata_files = {'README.md', 'readme.md', '.civitai.info', 'metadata.json'}
                if file_name in metadata_files or file_name.endswith('.civitai.info') or file_name.endswith('.metadata.json'):
                    if self.verbose or verbose_dry_run:
                        print(f"    📋 Skipping metadata file: {file_name}")
                    continue
                
                # Skip files that exceed max file size limit
                if self.config['max_file_size'] > 0 and file_size > self.config['max_file_size']:
                    if self.verbose or verbose_dry_run:
                        print(f"    ⏭️  Skipping large file: {file_name} ({file_size:,} bytes)")
                    continue
                
                # Show file processing in verbose mode
                if self.verbose or verbose_dry_run:
                    print(f"    📄 Processing file: {file_name} ({file_size:,} bytes)")
                
                # Process file based on type and mode
                file_result = self._process_single_file(file_path, file_size, fast_prescan, force_blake3, verbose_dry_run)
                
                if file_result:
                    # Categorize file
                    if file_path.lower().endswith(tuple(self.MODEL_EXTENSIONS)):
                        results['models'].append(file_result)
                        if self.verbose:
                            print(f"      🤖 Categorized as model: {file_name}")
                    elif file_path.lower().endswith(tuple(self.IMAGE_EXTENSIONS)):
                        results['images'].append(file_result)
                        if self.verbose:
                            print(f"      🖼️  Categorized as image: {file_name}")
                    elif file_path.lower().endswith(tuple(self.TEXT_EXTENSIONS)):
                        results['text_files'].append(file_result)
                        if self.verbose:
                            print(f"      📝 Categorized as text: {file_name}")
                    else:
                        results['other_files'].append(file_result)
                        if self.verbose:
                            print(f"      📦 Categorized as other: {file_name}")
                elif self.verbose:
                    print(f"      ❌ File skipped or failed processing: {file_name}")
        
        except Exception as e:
            print(f"Error scanning directory {directory}: {e}")
        
        return results

    def _check_and_mark_parent_completion(self, directory_path: str, folder_limit: Optional[int] = None):
        """Check if parent directory is now fully complete and mark it
        
        Args:
            directory_path: The directory that was just completed
            folder_limit: If set, be more conservative about parent completion during incremental scans
        """
        # Only mark the current directory as complete. Parent marking is handled separately after all children are verified.
        # This function is now a no-op or can be removed, but kept for compatibility.
    pass

    def scan_directory(self, directory: str, fast_prescan: bool = False, force_blake3: bool = False, folder_limit: Optional[int] = None, verbose_dry_run: bool = False) -> Dict[str, List[Dict]]:
        """
        Main directory scanning method - now uses hierarchical scanning by default
        
        Args:
            directory: Directory to scan
            fast_prescan: If True, only calculate AutoV3 hashes for fast pre-scanning
            force_blake3: If True, force BLAKE3 rescanning for non-SafeTensors files
            folder_limit: If specified, only process this many folders per run for incremental scanning
            verbose_dry_run: If True, show every file check and folder action (for dry-run mode)
        """
        # Use hierarchical scanning for better performance and organization
        return self.scan_directory_hierarchical(directory, fast_prescan, force_blake3, folder_limit, verbose_dry_run)

    def _check_directory_tables_exist(self) -> bool:
        """Check if directory tracking tables exist in the database (backward compatibility)"""
        try:
            cursor = self.db.conn.cursor()
            # Check if directory_scan_status table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='directory_scan_status'
            """)
            return cursor.fetchone() is not None
        except Exception:
            return False

    def _is_directory_complete_legacy(self, directory_path: str) -> bool:
        """
        Legacy method to check if directory is complete based on ALL files being fully scanned
        (checks if all files in directory exist in database and are properly processed)
        """
        try:
            # Get all files in this directory (non-recursive)
            files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
            
            if not files:
                return True  # Empty directories are considered complete
            
            # Check if ALL files are fully scanned in database
            fully_scanned_files = 0
            for file_name in files:
                file_path = os.path.join(directory_path, file_name)
                
                try:
                    file_size = os.path.getsize(file_path)
                    
                    # Skip very small files (less than 1KB - likely not real model files)
                    if file_size < 1024:
                        fully_scanned_files += 1  # Count tiny files as "complete"
                        continue
                    
                    # Check if file is fully scanned in database
                    cursor = self.db.conn.cursor()
                    cursor.execute("""
                        SELECT sf.id, sf.scan_status 
                        FROM scanned_files sf 
                        WHERE sf.file_name = ? AND sf.file_size = ? AND sf.file_path = ?
                    """, (file_name, file_size, file_path))
                    
                    result = cursor.fetchone()
                    if result:
                        file_id, scan_status = result
                        # File is complete if it has scan_status >= 2 (fully scanned)
                        if scan_status and scan_status >= 2:
                            fully_scanned_files += 1
                        # For files without scan_status, check if they have metadata
                        elif scan_status is None:
                            # Check if file has associated metadata (indicates it was processed)
                            cursor.execute("""
                                SELECT COUNT(*) FROM media_metadata 
                                WHERE scanned_file_id = ?
                            """, (file_id,))
                            if cursor.fetchone()[0] > 0:
                                fully_scanned_files += 1
                
                except Exception as e:
                    # If we can't stat the file or check database, assume incomplete
                    continue
            
            # Directory is complete ONLY if ALL files are fully scanned
            if len(files) == 0:
                return True
            
            is_complete = fully_scanned_files == len(files)
            
            if self.config.get('verbose', True) and len(files) > 0:
                print(f"    📊 Legacy check: {fully_scanned_files}/{len(files)} files fully scanned in {directory_path}")
            
            return is_complete
            
        except Exception as e:
            if self.config.get('verbose', True):
                print(f"    ⚠️ Error checking directory completion: {e}")
            return False  # If we can't verify, assume incomplete

    def _process_single_file(self, file_path: str, file_size: int, fast_prescan: bool = False, force_blake3: bool = False, verbose_dry_run: bool = False):
        """Process a single file and return file information"""
        try:
            file_name = os.path.basename(file_path)
            _, extension = os.path.splitext(file_name)
            extension = extension.lower()
            
            # Get file stats
            file_stat = os.stat(file_path)
            last_modified = file_stat.st_mtime
            
            # Skip very small files (likely corrupted or incomplete)
            if file_size < 1024:  # 1KB minimum
                return None
            
            if fast_prescan and not force_blake3:
                # ENHANCED FAST PRESCAN MODE: Check all files, add missing ones to database
                is_safetensors = file_path.lower().endswith('.safetensors')
                existing_record = None
                
                if is_safetensors:
                    # SafeTensors: Use AutoV3 for fast lookup
                    if self.verbose or self.config.get('verbose', False) or verbose_dry_run:
                        print(f"🔍 Checking SafeTensors file: {file_path}")
                    
                    autov3_hash = compute_autov3_hex(Path(file_path))
                    if autov3_hash:
                        autov3_hash = autov3_hash.upper()
                        existing_record = self.db.get_file_by_autov3(autov3_hash)
                else:
                    # Non-SafeTensors: Use BLAKE3 for fast lookup (if available)
                    if self.verbose or self.config.get('verbose', False) or verbose_dry_run:
                        print(f"🔍 Checking non-SafeTensors file: {file_path}")
                    
                    # Calculate BLAKE3 for fast lookup
                    if BLAKE3_AVAILABLE and blake3 is not None:
                        blake3_hasher = blake3.blake3()
                        with open(file_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(1048576), b""):
                                blake3_hasher.update(chunk)
                        blake3_hash = blake3_hasher.hexdigest().upper()
                        existing_record = self.db.get_file_by_blake3(blake3_hash)
                    
                    # Fall back to path lookup if no BLAKE3
                    if not existing_record:
                        existing_record = self.db.get_file_by_path(file_path)
                
                if existing_record:
                    # File already in database - return None (skip)
                    if self.config.get('verbose', False) or verbose_dry_run:
                        print(f"✅ File already in database: {file_path}")
                    return None
                
                # FILE NOT IN DATABASE: Fully scan and add it
                if self.config.get('verbose', False) or verbose_dry_run:
                    print(f"🆕 File not in database, fully scanning: {file_path}")
            
            # Calculate hashes for new/unscanned files
            sha256, autov3, blake3_hash = self.calculate_hashes(file_path)
            if not sha256:
                return None
            
            # Classify file type
            file_type = self.classify_file_type(file_path, extension)
            
            # Extract metadata for appropriate file types
            image_metadata = None
            comprehensive_metadata = None
            
            if file_type == 'image':
                image_metadata = extract_image_metadata(file_path)
                comprehensive_metadata = extract_comprehensive_metadata(file_path, image_metadata)
            
            # Add to database
            scanned_file_id = self.db.add_scanned_file(
                file_path, file_name, file_size, sha256, 
                file_type, extension, last_modified, autov3, blake3_hash, image_metadata
            )
            
            # Link file associations for newly scanned files
            if scanned_file_id:
                self.link_file_associations(scanned_file_id, file_path, file_type)
            
            # Add comprehensive metadata if extracted
            if comprehensive_metadata and scanned_file_id:
                try:
                    components = comprehensive_metadata.pop('components', [])
                    comprehensive_metadata['scan_status'] = 2  # Mark as fully scanned
                    
                    media_metadata_id = self.db.create_media_metadata(scanned_file_id, **comprehensive_metadata)
                    
                    # Store component usage
                    if media_metadata_id is not None:
                        for component in components:
                            self.db.add_component_usage(
                                media_metadata_id, 
                                component['type'], 
                                component['name'],
                                component.get('weight'),
                                component.get('hash'),
                                component.get('version'),
                                component.get('context')
                            )
                
                except Exception as e:
                    print(f"Warning: Could not save comprehensive metadata for {file_path}: {e}")
            
            # Handle batched commits
            self.batch_count += 1
            if self.batch_count >= self.batch_size:
                self.db.commit()
                self.batch_count = 0
            
            # Create file info dict
            file_info = {
                'id': scanned_file_id,
                'path': file_path,
                'name': file_name,
                'size': file_size,
                'sha256': sha256,
                'autov3': autov3,
                'blake3': blake3_hash,
                'type': file_type,
                'extension': extension,
                'last_modified': last_modified,
                'is_duplicate': False,
                'existing_file': None,
                'from_cache': False,
                'newly_added': True  # Flag to indicate this was just added
            }
            
            return file_info
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def rescan_missing_metadata(self) -> Dict:
        """Re-scan files that are missing metadata"""
        missing_files = self.db.get_files_missing_metadata()
        
        if not missing_files:
            print("All files have complete metadata.")
            return {'updated': 0, 'errors': 0, 'skipped': 0}
        
        print(f"Found {len(missing_files)} files with missing metadata...")
        
        stats = {'updated': 0, 'errors': 0, 'skipped': 0}
        
        for file_info in missing_files:
            file_path = file_info['file_path']
            missing_type = file_info['missing_metadata_type']
            
            # Check if file still exists
            if not os.path.exists(file_path):
                print(f"File no longer exists: {file_path}")
                stats['skipped'] += 1
                continue
            
            try:
                if missing_type == 'image':
                    # Extract image metadata
                    if self.config['verbose']:
                        print(f"Extracting image metadata: {file_path}")
                    
                    image_metadata = extract_image_metadata(file_path)
                    self.db.update_file_metadata(file_info['id'], image_metadata=image_metadata)
                    stats['updated'] += 1
                    
                elif missing_type == 'autov3':
                    # Extract AutoV3 hash
                    if self.config['verbose']:
                        print(f"Extracting AutoV3 hash: {file_path}")
                    
                    autov3_hash = compute_autov3_hex(Path(file_path))
                    if autov3_hash:
                        autov3_hash = autov3_hash.upper()
                    self.db.update_file_metadata(file_info['id'], autov3=autov3_hash)
                    stats['updated'] += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                stats['errors'] += 1
        
        print(f"\nMetadata re-scan complete:")
        print(f"  Files updated: {stats['updated']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Skipped: {stats['skipped']}")
        
        return stats
    
    def get_scan_summary(self) -> Dict:
        """Get summary of scanning results"""
        cursor = self.db.conn.cursor()
        
        # Get counts by file type
        cursor.execute('''
            SELECT file_type, COUNT(*) as count
            FROM scanned_files 
            GROUP BY file_type
        ''')
        type_counts = dict(cursor.fetchall())
        
        # Get total duplicates
        cursor.execute('''
            SELECT COUNT(*) FROM scanned_files sf1 
            WHERE EXISTS (
                SELECT 1 FROM scanned_files sf2 
                WHERE sf2.sha256 = sf1.sha256 AND sf2.id != sf1.id
            )
        ''')
        result = cursor.fetchone()
        duplicate_count = result[0] if result else 0        # Get recent operations
        cursor.execute('''
            SELECT operation_type, status, COUNT(*) as count
            FROM processing_log 
            WHERE created_at > ? 
            GROUP BY operation_type, status
        ''', (int(time.time()) - 3600,))  # Last hour
        recent_operations = cursor.fetchall()
        
        return {
            'total_files': sum(type_counts.values()),
            'by_type': type_counts,
            'duplicates_found': duplicate_count,
            'recent_operations': recent_operations
        }
    
    def scan_existing_files_for_metadata(self, limit: Optional[int] = 100, force_rescan: bool = False) -> Dict[str, int]:
        """Scan existing files in database to extract enhanced metadata"""
        stats = {
            'processed': 0,
            'updated': 0,
            'errors': 0,
            'skipped': 0
        }
        
        if limit is None:
            print(f"Scanning ALL existing files for enhanced metadata...")
        else:
            print(f"Scanning existing files for enhanced metadata (limit: {limit})...")
        
        # Get files that need metadata scanning
        if force_rescan:
            # For force rescan, get all media files
            cursor = self.db.conn.cursor()
            query = '''
                SELECT id, file_path, file_name, file_type 
                FROM scanned_files 
                WHERE file_type IN ('jpeg', 'jpg', 'png', 'webp', 'gif', 'bmp', 'tiff', 'image', 'video')
                ORDER BY id
            '''
            if limit:
                query += f' LIMIT {limit}'
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            files_to_scan = [dict(zip(columns, row)) for row in cursor.fetchall()]
        else:
            # Get only unscanned files using the new tracking
            files_to_scan = self.db.get_unscanned_media_files(limit)  # type: ignore

        if not files_to_scan:
            print("No files need metadata scanning")
            return stats

        print(f"Found {len(files_to_scan)} files needing metadata extraction")

        for file_info in files_to_scan:
            file_path = file_info['file_path']
            scanned_file_id = file_info['id']
            file_type = file_info['file_type']
            
            # Skip if already scanned and not forcing rescan
            if not force_rescan and self.db.is_media_metadata_scanned(scanned_file_id):  # type: ignore
                stats['skipped'] += 1
                continue            # Check if file still exists with better error reporting and recovery
            if not os.path.exists(file_path):
                # Attempt to recover the missing file
                recovery_success, recovery_result = self.recover_missing_file(file_path, scanned_file_id)
                
                if recovery_success:
                    new_file_path = recovery_result
                    print(f"🔄 File recovered: {os.path.basename(file_path)}")
                    print(f"   Old location: {file_path}")
                    print(f"   New location: {new_file_path}")
                    
                    # Update the database with the new path
                    cursor = self.db.conn.cursor()
                    cursor.execute('UPDATE scanned_files SET file_path = ? WHERE id = ?', 
                                 (new_file_path, scanned_file_id))
                    self.db.conn.commit()
                    
                    # Update file_path for continued processing
                    file_path = new_file_path
                    file_info['file_path'] = new_file_path
                    
                else:
                    # Recovery failed, show detailed error information
                    if self.config['verbose']:
                        print(f"File not found at expected location: {file_path}")
                        print(f"   Recovery failed: {recovery_result}")
                        # Check if it's a permission issue or truly missing
                        parent_dir = os.path.dirname(file_path)
                        if os.path.exists(parent_dir):
                            print(f"  Parent directory exists: {parent_dir}")
                            try:
                                files_in_dir = os.listdir(parent_dir)
                                filename = os.path.basename(file_path)
                                if filename in files_in_dir:
                                    print(f"  File exists in directory but access failed - permissions issue?")
                                else:
                                    print(f"  File not in directory (may have been moved/deleted)")
                                    if len(files_in_dir) <= 5:  # Show files if directory is small
                                        print(f"  Files in directory: {files_in_dir}")
                            except PermissionError:
                                print(f"  Permission denied accessing directory: {parent_dir}")
                        else:
                            print(f"  Parent directory also missing: {parent_dir}")
                    else:
                        print(f"File no longer exists: {file_path}")
                        
                    # Mark as not found
                    self.db.mark_media_metadata_scanned(scanned_file_id, success=False, scan_notes="File not found")  # type: ignore
                    stats['skipped'] += 1
                    continue
            
            try:
                civitai_metadata = None
                
                # Extract metadata based on file type
                comprehensive_metadata = None
                if file_type == 'image':
                    if self.config['verbose']:
                        print(f"Extracting metadata from image: {file_path}")
                    
                    # Extract basic image metadata first
                    image_metadata = extract_image_metadata(file_path)
                    
                    # Extract comprehensive metadata
                    comprehensive_metadata = extract_comprehensive_metadata(file_path, image_metadata)
                
                elif file_type == 'video' or (file_type == 'other' and 
                     os.path.splitext(file_path)[1].lower() in ['.mp4', '.webm', '.mov', '.avi']):
                    if self.config['verbose']:
                        print(f"Extracting metadata from video: {file_path}")
                    
                    # Extract basic metadata for videos
                    comprehensive_metadata = extract_comprehensive_metadata(file_path, {})
                
                # Update or create metadata record
                if comprehensive_metadata:
                    # Extract components before storing main metadata
                    components = comprehensive_metadata.pop('components', [])
                    comprehensive_metadata['scan_status'] = 1  # Mark as successfully scanned
                    
                    # Check if metadata record exists
                    existing_metadata = self.db.get_media_metadata(scanned_file_id)
                    
                    if existing_metadata:
                        # Update existing record
                        self.db.update_media_metadata(scanned_file_id, **comprehensive_metadata)
                        media_metadata_id = existing_metadata['id']
                    else:
                        # Create new record
                        media_metadata_id = self.db.create_media_metadata(scanned_file_id, **comprehensive_metadata)
                    
                    # Store component usage (clear existing first if updating) - only if we have a valid ID
                    if media_metadata_id is not None:
                        if existing_metadata and components:
                            # Clear existing components for this media
                            cursor = self.db.conn.cursor()
                            cursor.execute('DELETE FROM component_usage WHERE media_metadata_id = ?', (media_metadata_id,))
                        
                        # Add new component usage records
                        for component in components:
                            self.db.add_component_usage(
                                media_metadata_id, 
                                component['type'], 
                                component['name'],
                                component.get('weight'),
                                component.get('hash'),
                                component.get('version'),
                                component.get('context')
                            )
                    
                    # Mark as successfully scanned
                    self.db.mark_media_metadata_scanned(scanned_file_id, success=True, scan_notes="Metadata extracted successfully")  # type: ignore
                    
                    stats['updated'] += 1
                    if self.config['verbose']:
                        comp_info = f" ({len(components)} components)" if components else ""
                        print(f"Updated metadata for: {os.path.basename(file_path)}{comp_info}")
                else:
                    # No metadata found, mark as scanned but with no data
                    self.db.mark_media_metadata_scanned(scanned_file_id, success=True, scan_notes="No extractable metadata found")  # type: ignore
                    stats['skipped'] += 1
                
                stats['processed'] += 1
                
                # Commit periodically
                if stats['processed'] % 50 == 0:
                    self.db.commit()
                    print(f"Processed {stats['processed']} files...")
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                log_metadata_error(f"Error processing file: {e}", file_path, str(e))
                self.db.mark_media_metadata_scanned(scanned_file_id, success=False, scan_notes=f"Error: {str(e)}")  # type: ignore
                self.db.log_operation('metadata_extraction', file_path, 'error', str(e))
                stats['errors'] += 1
        
        # Final commit
        self.db.commit()
        
        # Save column lacking errors for retry functionality  
        self._save_column_lacking_errors()
        
        print(f"Metadata scanning complete: {stats['processed']} processed, "
              f"{stats['updated']} updated, {stats['errors']} errors, {stats['skipped']} skipped")
        
        return stats
    
    def retry_failed_metadata_extraction(self) -> Dict[str, int]:
        """Retry metadata extraction for files that previously failed"""
        
        # Load previously failed files from both sources
        failed_files = self._load_column_lacking_errors()
        error_log_files = self._load_error_log_failures()
        
        # Combine failed files from both sources, avoiding duplicates
        all_failed_files = {}
        
        # Add column lacking errors
        for error_entry in failed_files:
            file_path = error_entry.get('file_path')
            if file_path:
                all_failed_files[file_path] = error_entry
        
        # Add error log failures  
        for error_entry in error_log_files:
            file_path = error_entry.get('file_path')
            if file_path and file_path not in all_failed_files:
                all_failed_files[file_path] = error_entry
        
        if not all_failed_files:
            print("No previously failed files found to retry.")
            return {'processed': 0, 'updated': 0, 'errors': 0, 'skipped': 0}
        
        print(f"Found {len(failed_files)} files with missing column errors")
        print(f"Found {len(error_log_files)} files with parsing errors")
        print(f"Total unique failed files to retry: {len(all_failed_files)}")
        print("Retrying metadata extraction...")
        
        stats = {'processed': 0, 'updated': 0, 'errors': 0, 'skipped': 0}
        
        # Clear current session errors to track new failures
        self.column_lacking_errors = []
        
        for file_path, error_entry in all_failed_files.items():
            if isinstance(error_entry, dict):
                file_path = error_entry['file_path']
            
            if not os.path.exists(file_path):
                print(f"File no longer exists: {file_path}")
                stats['skipped'] += 1
                continue
                
            try:
                # Get file info from database
                file_info = self.db.get_file_by_path(file_path)
                if not file_info:
                    print(f"File not found in database: {file_path}")
                    stats['skipped'] += 1
                    continue
                
                scanned_file_id = file_info['id']
                
                # Extract metadata using the enhanced extraction
                basic_metadata = {
                    'file_path': file_path,
                    'file_size': file_info['file_size'],
                    'last_modified': file_info['last_modified']
                }
                metadata = extract_comprehensive_metadata(file_path, basic_metadata)
                
                if metadata:
                    # Try to update/create metadata record
                    existing_metadata = self.db.get_media_metadata(scanned_file_id)
                    
                    # Remove components for separate handling
                    components = metadata.pop('components', [])
                    metadata['scan_status'] = 1  # Mark as successfully scanned
                    
                    if existing_metadata:
                        self.db.update_media_metadata(scanned_file_id, **metadata)
                    else:
                        metadata['scanned_file_id'] = scanned_file_id
                        self.db.create_media_metadata(**metadata)
                    
                    # Store components separately
                    for component in components:
                        self.db.add_component_usage(
                            scanned_file_id,
                            component['type'],
                            component['name'],
                            component.get('weight'),
                            component.get('hash'),
                            component.get('version'),
                            component.get('context')
                        )
                    
                    print(f"✅ Successfully processed: {os.path.basename(file_path)}")
                    stats['updated'] += 1
                    
                    # Update retry count in error entry
                    error_entry['retry_count'] = error_entry.get('retry_count', 0) + 1
                    error_entry['last_retry'] = __import__('datetime').datetime.now().isoformat()
                    
                else:
                    stats['skipped'] += 1
                
                stats['processed'] += 1
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Still failing: {os.path.basename(file_path)} - {error_msg}")
                stats['errors'] += 1
                
                # Track if it's still a column lacking error
                if "no column named" in error_msg:
                    self._track_column_lacking_error(file_path, error_msg)
        
        # Final commit
        self.db.commit()
        
        # Save any new column lacking errors
        self._save_column_lacking_errors()
        
        print(f"\nRetry complete: {stats['processed']} processed, "
              f"{stats['updated']} updated, {stats['errors']} errors, {stats['skipped']} skipped")
        
        return stats
    
    def process_orphaned_media(self, media_files: list, dry_run: bool = False) -> Dict[str, int]:
        """Process orphaned media files using cross-referencing and intelligent sorting"""
        stats = {
            'processed': 0,
            'moved_to_model': 0,
            'moved_to_duplicates': 0,
            'ignored': 0,
            'errors': 0
        }
        
        print(f"Processing {len(media_files)} orphaned media files...")
        
        for media_info in media_files:
            try:
                media_path = media_info['file_path']  # Use 'file_path' from get_orphaned_media_files
                media_name = media_info['file_name']  # Use 'file_name' from get_orphaned_media_files
                media_id = media_info['id']
                
                if self.config['verbose']:
                    print(f"Processing orphaned media: {media_name}")
                
                # Step 1: Check if media file has embedded metadata
                existing_metadata = self.db.get_media_metadata(media_id)
                target_model_info = None
                
                if existing_metadata and existing_metadata.get('model_hash'):
                    # Media has model hash in metadata - find the model
                    target_model_info = self.db.find_model_by_hash(existing_metadata['model_hash'])
                
                # Step 2: If no embedded metadata, look for metadata text file
                if not target_model_info:
                    metadata_file = find_metadata_text_file(media_path)
                    if metadata_file:
                        if self.config['verbose']:
                            print(f"Found metadata file: {os.path.basename(metadata_file)}")
                        
                        text_metadata = read_metadata_text_file(metadata_file)
                        
                        # Look for model references in text file
                        for ref in text_metadata.get('model_references', []):
                            if ref['type'] == 'hash':
                                target_model_info = self.db.find_model_by_hash(ref['value'])
                                if target_model_info:
                                    break
                            elif ref['type'] == 'name':
                                # Try to find model by name (less reliable)
                                models = self.db.find_component_by_name(ref['value'], 'model')
                                if models:
                                    target_model_info = models[0]
                                    break
                        
                        # Look for LoRA/component references
                        if not target_model_info:
                            for component in text_metadata.get('found_components', []):
                                # Try to find the component file
                                components = self.db.find_component_by_name(component['name'], component['type'])
                                if components:
                                    # Use the component's directory as target
                                    target_model_info = components[0]
                                    break
                
                # Step 3: If still no match, try civitai cross-reference
                if not target_model_info:
                    sha256_hash = media_info.get('sha256', '')
                    blur_hash = existing_metadata.get('blur_hash') if existing_metadata else None
                    
                    if sha256_hash:
                        civitai_results = cross_reference_with_civitai(
                            sha256_hash, 
                            blur_hash or '', 
                            self.config.get('database_path', 'Database/civitai.sqlite')  # Use config path for civitai database
                        )
                        
                        if civitai_results['civitai_matches']:
                            if self.config['verbose']:
                                print(f"Found {len(civitai_results['civitai_matches'])} Civitai matches")
                            
                            # Try to match civitai result with local models
                            for match in civitai_results['civitai_matches']:
                                # Look for local model with matching name or base_model
                                local_models = self.db.find_component_by_name(match['name'])
                                if local_models:
                                    target_model_info = local_models[0]
                                    break
                                else:
                                    # Use civitai data directly, but normalize field names
                                    target_model_info = {
                                        'component_name': match['name'],
                                        'component_type': match.get('model_type', 'other'),
                                        'base_model': match.get('base_model', 'unknown'),
                                        'file_name': match['name']
                                    }
                
                # Step 4: Move file to appropriate location
                if target_model_info:
                    target_dir = self._get_model_directory(target_model_info)
                    target_path = os.path.join(target_dir, media_name)
                    
                    # Check for duplicates
                    if os.path.exists(target_path):
                        # Move to duplicates folder
                        duplicate_dir = self._get_duplicate_media_directory(target_model_info)
                        duplicate_path = os.path.join(duplicate_dir, media_name)
                        
                        if not dry_run:
                            os.makedirs(duplicate_dir, exist_ok=True)
                            if not os.path.exists(duplicate_path):
                                shutil.move(media_path, duplicate_path)
                                self.db.update_scanned_file_location(media_id, duplicate_path)
                        
                        stats['moved_to_duplicates'] += 1
                        if self.config['verbose']:
                            print(f"Moved to duplicates: {duplicate_path}")
                    else:
                        # Move to model directory
                        if not dry_run:
                            os.makedirs(target_dir, exist_ok=True)
                            shutil.move(media_path, target_path)
                            self.db.update_scanned_file_location(media_id, target_path)
                        
                        stats['moved_to_model'] += 1
                        if self.config['verbose']:
                            print(f"Moved to model directory: {target_path}")
                else:
                    # Step 5: No match found - mark as ignored
                    self.db.mark_media_ignored(media_id, "No model cross-reference found")
                    stats['ignored'] += 1
                    if self.config['verbose']:
                        print(f"Marked as ignored: {media_name}")
                
                stats['processed'] += 1
                
            except Exception as e:
                print(f"Error processing media {media_info.get('name', 'unknown')}: {e}")
                stats['errors'] += 1
        
        return stats
    
    def _get_model_directory(self, model_info: Dict) -> str:
        """Get the target directory for a model based on its type and base model"""
        base_dest = self.config['destination_directory']
        
        model_type = model_info.get('component_type', model_info.get('file_type', 'other'))
        base_model = model_info.get('base_model', 'unknown')
        model_name = model_info.get('component_name', model_info.get('file_name', ''))
        
        # Create directory structure: models/checkpoints/SD 1.5/model_name/
        if self.config.get('use_model_type_subfolders', True):
            type_folder = f"{model_type}s" if not model_type.endswith('s') else model_type
            return os.path.join(base_dest, type_folder, base_model, model_name)
        else:
            return os.path.join(base_dest, base_model, model_name)
    
    def _get_duplicate_media_directory(self, model_info: Dict) -> str:
        """Get the duplicate media directory based on model type"""
        base_dest = self.config['destination_directory']
        model_type = model_info.get('component_type', model_info.get('file_type', 'other'))
        
        return os.path.join(base_dest, 'duplicates', 'media', model_type)
    
    def get_orphaned_media_files(self) -> list:
        """Get media files that are not associated with any model"""
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT sf.id, sf.file_path, sf.file_name, sf.sha256, sf.file_type
            FROM scanned_files sf
            LEFT JOIN media_metadata mm ON sf.id = mm.scanned_file_id
            WHERE sf.file_type IN ('image', 'video')
              AND sf.file_path NOT LIKE '%/models/%'  -- Not already in sorted structure
              AND (mm.scan_status IS NULL OR mm.scan_status != -1)  -- Not ignored
              AND NOT (sf.file_path LIKE '%/lora%' AND sf.file_path LIKE '%model%')  -- Not next to models
            LIMIT 50  -- Limit for testing
        ''')
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_unprocessed_media_files(self) -> list:
        """Get media files that haven't had metadata extracted"""
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT sf.id, sf.file_path, sf.file_name, sf.file_type
            FROM scanned_files sf
            LEFT JOIN media_metadata mm ON sf.id = mm.scanned_file_id
            WHERE sf.file_type IN ('image', 'video')
              AND mm.scanned_file_id IS NULL  -- No metadata record exists
        ''')
        
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def scan_metadata_incrementally(self, file_id: int, metadata_fields) -> Dict[str, Any]:
        """
        Incrementally scan metadata for a file based on field selections
        Only extracts metadata for fields that are selected and not already scanned
        
        Args:
            file_id: Database ID of the scanned file
            metadata_fields: Dict of {field_name: should_scan_boolean} or list of field names
        
        Returns:
            Dict containing scan results and extracted metadata
        """
        results = {
            'scanned_fields': [],
            'skipped_fields': [],
            'failed_fields': [],
            'extracted_metadata': {}
        }
        
        # Convert list to dictionary if needed
        if isinstance(metadata_fields, list):
            metadata_checkboxes = {field: True for field in metadata_fields}
        elif isinstance(metadata_fields, dict):
            metadata_checkboxes = metadata_fields
        else:
            results['error'] = f'metadata_fields must be a dict or list, got {type(metadata_fields)}'
            return results
        
        # Get file info
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT file_path, file_type FROM scanned_files WHERE id = ?', (file_id,))
        file_info = cursor.fetchone()
        
        if not file_info:
            results['error'] = f'File ID {file_id} not found in database'
            return results
        
        file_path, file_type = file_info
        
        # Get already scanned metadata status
        cursor.execute('''
            SELECT field_name, scan_status FROM metadata_scan_status 
            WHERE scanned_file_id = ?
        ''', (file_id,))
        
        existing_scans = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Process each checkbox
        for field_name, should_scan in metadata_checkboxes.items():
            if not should_scan:
                results['skipped_fields'].append(f'{field_name}: Not selected')
                continue
            
            # Check if already successfully scanned (status = 1)
            if existing_scans.get(field_name) == 1:
                results['skipped_fields'].append(f'{field_name}: Already scanned')
                continue
            
            # Attempt to extract this specific metadata field
            try:
                field_value = self._extract_specific_metadata_field(file_path, file_type, field_name)
                
                if field_value is not None:
                    # Store in database
                    cursor.execute('''
                        INSERT OR REPLACE INTO metadata_scan_status 
                        (scanned_file_id, field_name, scan_status, field_value, scan_date)
                        VALUES (?, ?, 1, ?, strftime('%s', 'now'))
                    ''', (file_id, field_name, str(field_value)))
                    
                    results['scanned_fields'].append(f'{field_name}: {field_value}')
                    results['extracted_metadata'][field_name] = field_value
                else:
                    # Mark as not applicable/not found (status = 3)
                    cursor.execute('''
                        INSERT OR REPLACE INTO metadata_scan_status 
                        (scanned_file_id, field_name, scan_status, scan_notes, scan_date)
                        VALUES (?, ?, 3, 'Field not found or not applicable', strftime('%s', 'now'))
                    ''', (file_id, field_name))
                    
                    results['skipped_fields'].append(f'{field_name}: Not found')
                
            except Exception as e:
                # Mark as failed (status = 2)
                cursor.execute('''
                    INSERT OR REPLACE INTO metadata_scan_status 
                    (scanned_file_id, field_name, scan_status, scan_notes, scan_date)
                    VALUES (?, ?, 2, ?, strftime('%s', 'now'))
                ''', (file_id, field_name, str(e)))
                
                results['failed_fields'].append(f'{field_name}: {str(e)}')
        
        self.db.conn.commit()
        return results
    
    def _extract_specific_metadata_field(self, file_path: str, file_type: str, field_name: str):
        """Extract a specific metadata field from a file"""
        
        # Only process image files for metadata extraction
        if file_type != 'image':
            return None
        
        # Get basic image metadata first
        basic_metadata = extract_image_metadata(file_path)
        
        # Get comprehensive metadata
        comprehensive_metadata = extract_comprehensive_metadata(file_path, basic_metadata)
        
        # Field mapping
        field_mappings = {
            'width': lambda m: basic_metadata.get('width'),
            'height': lambda m: basic_metadata.get('height'),
            'steps': lambda m: comprehensive_metadata.get('steps'),
            'sampler': lambda m: comprehensive_metadata.get('sampler'),
            'cfg_scale': lambda m: comprehensive_metadata.get('cfg_scale'),
            'seed': lambda m: comprehensive_metadata.get('seed'),
            'model_name': lambda m: comprehensive_metadata.get('model_name'),
            'model_hash': lambda m: comprehensive_metadata.get('model_hash'),
            'generation_tool': lambda m: comprehensive_metadata.get('generation_tool'),
            'prompt_text': lambda m: comprehensive_metadata.get('prompt_text'),
            'negative_prompt': lambda m: comprehensive_metadata.get('negative_prompt'),
            'vae_name': lambda m: comprehensive_metadata.get('vae_name'),
            'vae_hash': lambda m: comprehensive_metadata.get('vae_hash'),
            'clip_skip': lambda m: comprehensive_metadata.get('clip_skip'),
            'denoising_strength': lambda m: comprehensive_metadata.get('denoising_strength'),
            'hires_upscaler': lambda m: comprehensive_metadata.get('hires_upscaler'),
            'hires_steps': lambda m: comprehensive_metadata.get('hires_steps'),
            'hires_upscale': lambda m: comprehensive_metadata.get('hires_upscale'),
            'has_components': lambda m: len(comprehensive_metadata.get('components', [])) > 0,
            'component_count': lambda m: len(comprehensive_metadata.get('components', [])),
            'civitai_id': lambda m: comprehensive_metadata.get('civitai_id'),
            'civitai_uuid': lambda m: comprehensive_metadata.get('civitai_uuid'),
            'blur_hash': lambda m: comprehensive_metadata.get('blur_hash'),
            'nsfw_level': lambda m: comprehensive_metadata.get('nsfw_level')
        }
        
        # Extract the requested field
        if field_name in field_mappings:
            return field_mappings[field_name](comprehensive_metadata)
        
        return None
    
    def get_available_metadata_fields(self) -> Dict[str, str]:
        """Get list of available metadata fields with descriptions"""
        return {
            'width': 'Image width in pixels',
            'height': 'Image height in pixels', 
            'steps': 'Generation steps (Automatic1111/ComfyUI)',
            'sampler': 'Sampler used (Euler a, DPM++ 2M, etc.)',
            'cfg_scale': 'CFG Scale value',
            'seed': 'Generation seed',
            'model_name': 'Base model name',
            'model_hash': 'Model hash (first 10 chars)',
            'generation_tool': 'Tool used (Automatic1111, ComfyUI, etc.)',
            'prompt_text': 'Positive prompt text',
            'negative_prompt': 'Negative prompt text',
            'vae_name': 'VAE model name',
            'vae_hash': 'VAE model hash',
            'clip_skip': 'CLIP skip value',
            'denoising_strength': 'Denoising strength (img2img)',
            'hires_upscaler': 'Hires upscaler name',
            'hires_steps': 'Hires steps',
            'hires_upscale': 'Hires upscale factor',
            'has_components': 'Has LoRA/LyCO components',
            'component_count': 'Number of components used',
            'civitai_id': 'Civitai ID (if available)',
            'civitai_uuid': 'Civitai UUID (if available)', 
            'blur_hash': 'Visual similarity hash',
            'nsfw_level': 'NSFW rating (if available)'
        }
    
    def get_metadata_scan_report(self, file_id: Optional[int] = None) -> Dict:
        """Get a report of metadata scan status for files"""
        cursor = self.db.conn.cursor()
        
        if file_id:
            # Report for specific file
            cursor.execute('''
                SELECT sf.file_name, sf.file_type, sf.file_path,
                       mss.field_name, mss.scan_status, mss.field_value, mss.scan_notes
                FROM scanned_files sf
                LEFT JOIN metadata_scan_status mss ON sf.id = mss.scanned_file_id
                WHERE sf.id = ?
                ORDER BY mss.field_name
            ''', (file_id,))
        else:
            # Summary report for all files
            cursor.execute('''
                SELECT field_name, 
                       COUNT(CASE WHEN scan_status = 1 THEN 1 END) as successful,
                       COUNT(CASE WHEN scan_status = 2 THEN 1 END) as failed,
                       COUNT(CASE WHEN scan_status = 3 THEN 1 END) as not_applicable,
                       COUNT(*) as total_attempts
                FROM metadata_scan_status
                GROUP BY field_name
                ORDER BY field_name
            ''')
        
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return {
            'report_type': 'file_specific' if file_id else 'summary',
            'file_id': file_id,
            'results': results
        }
    
    def force_blake3_scan(self, directory: str) -> Dict[str, int]:
        """Efficiently add BLAKE3 hashes to existing database files in directory"""
        stats = {
            'total_found': 0,
            'files_processed': 0,
            'files_updated': 0,
            'files_skipped': 0,
            'errors': 0
        }
        
        print(f"Scanning directory for non-SafeTensors files: {directory}")
        
        # Get files that need BLAKE3 hashes from database (much more efficient!)
        files_needing_blake3 = self.db.get_files_needing_blake3(directory)
        stats['total_found'] = len(files_needing_blake3)
        
        if stats['total_found'] == 0:
            print("No files need BLAKE3 hashes in this directory")
            return stats
        
        print(f"Found {stats['total_found']} files needing BLAKE3 hashes")
        
        for file_record in files_needing_blake3:
            file_path = file_record['file_path']
            file_id = file_record['id']
            
            try:
                # Check if file still exists
                if not os.path.exists(file_path):
                    if self.config['verbose']:
                        print(f"File no longer exists, skipping: {file_path}")
                    stats['files_skipped'] += 1
                    continue
                
                if self.config['verbose']:
                    print(f"Adding BLAKE3 hash: {file_path}")
                
                # Calculate only BLAKE3 hash (ultra-fast!)
                _, _, blake3_hash = self.calculate_hashes(file_path, force_blake3=True)
                
                if self.config['verbose']:
                    print(f"BLAKE3 hash result: {blake3_hash}")
                
                if blake3_hash:
                    # Update database
                    self.db.update_blake3_hash(file_id, blake3_hash)
                    stats['files_updated'] += 1
                    
                    # Handle batched commits
                    self.batch_count += 1
                    if self.batch_count >= self.batch_size:
                        self.db.commit()
                        self.batch_count = 0
                else:
                    if self.config['verbose']:
                        print(f"Could not calculate BLAKE3 hash for: {file_path}")
                    stats['errors'] += 1
                
                stats['files_processed'] += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                stats['errors'] += 1
        
        # Final commit for remaining files
        if self.batch_count > 0:
            self.db.commit()
            if self.config['verbose']:
                print(f"Final database commit for {self.batch_count} remaining files")
        
        return stats
    
    def close(self):
        """Clean up resources"""
        self.save_hash_cache()
        self.db.close()

def get_default_config():
    """Get default configuration"""
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'source_directory': '.',
        'destination_directory': './models',
        'database_path': 'Database/file_scanner.sqlite',
        'verbose': 'false',
        'max_file_size': '10737418240',  # 10GB
        'use_model_type_subfolders': 'true'
    }
    return config


def main():
    """Main entry point with comprehensive processing options"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Stable Diffusion Model Organizer with Advanced Cross-Referencing")
    parser.add_argument("--scan-directory", help="Directory to scan for models and media")
    parser.add_argument("--config", default="config.ini", help="Configuration file path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without moving files")
    parser.add_argument("--force-rescan", action="store_true", help="Force re-scan of all files, ignoring cache")
    parser.add_argument("--process-orphaned", action="store_true", help="Process orphaned media files using cross-referencing")
    parser.add_argument("--sort-models", action="store_true", help="Sort model files into organized directory structure")
    parser.add_argument("--extract-metadata", action="store_true", help="Extract and analyze metadata from media files")
    parser.add_argument("--retry-failed", action="store_true", help="Retry metadata extraction for files that previously failed due to missing columns")
    parser.add_argument("--migrate-paths", nargs=2, metavar=('OLD_PREFIX', 'NEW_PREFIX'), help="Migrate database paths from old prefix to new prefix (e.g., '/mnt/user/' '/mnt/')")
    parser.add_argument("--recover-missing", action="store_true", help="Automatically recover missing files by checking alternative paths and destinations")
    parser.add_argument("--repair-text-paths", action="store_true", help="Scan filesystem and repair database path mismatches for text/metadata files")
    parser.add_argument("--enhance-models", action="store_true", help="Enhance model records with civitai data and pattern-based base model detection")
    parser.add_argument("--enhance-models-direct", nargs='?', const='Database/civitai.sqlite', help="Directly enhance Unknown/Other models from collection database (default: configured database)")
    parser.add_argument("--apply-enhancements", action="store_true", help="Apply enhancement suggestions to database (use with --enhance-models-direct)")
    parser.add_argument("--dry-run-migrate", action="store_true", help="Show what path migration would do without making changes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not any([args.scan_directory, args.process_orphaned, args.sort_models, args.extract_metadata, args.retry_failed, args.migrate_paths, args.recover_missing, args.repair_text_paths, args.enhance_models, args.enhance_models_direct]):
        parser.print_help()
        print("\nError: You must specify at least one operation:")
        print("  --scan-directory DIR    # Scan for new files")
        print("  --process-orphaned      # Process orphaned media using cross-referencing")
        print("  --sort-models          # Sort model files into organized structure")
        print("  --extract-metadata     # Extract metadata from media files")
        print("  --retry-failed         # Retry files that previously failed due to missing columns")
        print("  --migrate-paths OLD NEW # Migrate database file paths from old prefix to new")
        print("  --recover-missing      # Automatically recover missing files")
        print("  --repair-text-paths    # Scan filesystem and repair database path mismatches for text/metadata files")
        print("  --enhance-models       # Enhance model records with civitai data and pattern-based base model detection")
        print("  --enhance-models-direct [DB_PATH] # Directly enhance Unknown/Other models from collection database")
        return 1
    
    # Load configuration
    config = configparser.ConfigParser()
    if os.path.exists(args.config):
        config.read(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        print(f"Warning: Configuration file {args.config} not found. Using defaults.")
        config = get_default_config()
    
    # Override verbose setting from command line
    if args.verbose:
        config['DEFAULT']['verbose'] = 'true'
    
    # Initialize scanner
    scanner = FileScanner(args.config)
    
    try:
        total_stats = {
            'total_files': 0,
            'new_files': 0,
            'cached_files': 0,
            'models_found': 0,
            'media_processed': 0,
            'orphaned_processed': 0,
            'models_sorted': 0,
            'metadata_extracted': 0
        }
        
        # Step 1: Scan directory for new files
        if args.scan_directory:
            if not os.path.exists(args.scan_directory):
                print(f"Error: Scan directory '{args.scan_directory}' does not exist.")
                return 1
            
            print(f"\n🔍 Scanning directory: {args.scan_directory}")
            results = scanner.scan_directory(args.scan_directory)
            
            # Extract counts from results
            models_count = len(results.get('models', []))
            images_count = len(results.get('images', []))
            total_files = models_count + images_count + len(results.get('text_files', [])) + len(results.get('other_files', []))
            
            total_stats['total_files'] = total_files
            total_stats['models_found'] = models_count
            total_stats['media_processed'] = images_count
            
            print(f"✅ Scan completed. Files processed: {total_files}")
            print(f"   Models detected: {models_count}")
            print(f"   Media files found: {images_count}")
            print(f"   Text files: {len(results.get('text_files', []))}")
            print(f"   Other files: {len(results.get('other_files', []))}")
        
        # Step 2: Extract metadata from media files
        if args.extract_metadata:
            print(f"\n🔍 Extracting metadata from media files...")
            print("Metadata extraction feature requires additional development.")
            print("Use --process-orphaned to process existing media files with cross-referencing.")
        
        # Step 2.5: Retry failed metadata extraction
        if args.retry_failed:
            print(f"\n🔄 Retrying failed metadata extraction...")
            stats = scanner.retry_failed_metadata_extraction()
            
            print(f"✅ Retry completed:")
            print(f"   Processed: {stats['processed']}")
            print(f"   Updated: {stats['updated']}")
            print(f"   Errors: {stats['errors']}")
            print(f"   Skipped: {stats['skipped']}")
        
        # Step 2.6: Migrate database paths
        if args.migrate_paths:
            old_prefix, new_prefix = args.migrate_paths
            dry_run_migrate = args.dry_run_migrate or args.dry_run
            
            print(f"\n🔄 Migrating database paths...")
            print(f"   From: {old_prefix}")
            print(f"   To:   {new_prefix}")
            print(f"   Mode: {'Dry run' if dry_run_migrate else 'Live update'}")
            
            results = scanner.migrate_database_paths(old_prefix, new_prefix, dry_run=dry_run_migrate)
            
            print(f"\n✅ Path migration results:")
            print(f"   Files found with old prefix: {results['files_found']}")
            print(f"   Files verified at new path: {results['files_verified']}")
            print(f"   Files missing at new path: {results['files_missing']}")
            if not dry_run_migrate:
                print(f"   Files updated in database: {results['files_updated']}")
            
            if results['files_missing'] > 0:
                print(f"\n⚠️  Warning: {results['files_missing']} files not found at new paths")
                print("   Consider checking if files were actually moved/renamed")
        
        # Step 2.7: Recover missing files
        if args.recover_missing:
            print(f"\n🔄 Recovering missing files...")
            recovery_results = scanner.bulk_recover_missing_files(dry_run=args.dry_run)
            
            print(f"\n✅ File recovery results:")
            print(f"   Files checked: {recovery_results['total_checked']}")
            print(f"   Files recovered: {recovery_results['recovered']}")
            print(f"   Files still missing: {recovery_results['still_missing']}")
            print(f"   Database updates: {recovery_results['db_updates']}")
        
        # Step 2.8: Repair text file paths
        if args.repair_text_paths:
            print(f"\n🔧 Repairing text file paths in database...")
            repair_results = scanner.rescan_and_repair_text_file_paths()
            
            print(f"\n✅ Text file path repair results:")
            print(f"   Associated files checked: {repair_results.get('files_checked', 0)}")
            print(f"   Paths corrected: {repair_results.get('paths_corrected', 0)}")
            print(f"   Models enhanced from civitai: {repair_results.get('models_enhanced', 0)}")
            print(f"   Base models detected from patterns: {repair_results.get('base_models_detected', 0)}")
            print(f"   Files not found: {repair_results.get('files_missing', 0)}")
            print(f"   Errors: {repair_results.get('errors', 0)}")
            
            if repair_results.get('errors', 0) > 0:
                print(f"   Error details saved to: wrong_path.json")
        
        # Step 2.9: Enhance model records
        if args.enhance_models:
            print(f"\n🔧 Enhancing model records with comprehensive detection...")
            enhancement_results = scanner.enhance_and_repair_model_records()
            
            print(f"\n✅ Model enhancement results:")
            print(f"   Models checked: {enhancement_results.get('models_checked', 0)}")
            print(f"   Paths corrected: {enhancement_results.get('paths_corrected', 0)}")
            print(f"   Models enhanced from civitai: {enhancement_results.get('models_enhanced_civitai', 0)}")
            print(f"   Base models detected from patterns: {enhancement_results.get('base_models_detected', 0)}")
            print(f"   Models missing from filesystem: {enhancement_results.get('models_missing', 0)}")
            print(f"   Errors: {enhancement_results.get('errors', 0)}")
        
        # Step 2.10: Direct enhance model records from collection database
        if args.enhance_models_direct:
            print(f"\n🎯 Directly enhancing Unknown/Other models from collection database: {args.enhance_models_direct}")
            enhancement_results = scanner.enhance_models_direct(args.enhance_models_direct, apply_updates=args.apply_enhancements, verbose=args.verbose)
            
            print(f"\n✅ Direct enhancement results:")
            print(f"   Unknown models checked: {enhancement_results.get('unknown_models_checked', 0)}")
            print(f"   Other models checked: {enhancement_results.get('other_models_checked', 0)}")
            print(f"   Models enhanced from civitai: {enhancement_results.get('models_enhanced_civitai', 0)}")
            print(f"   Base models detected from patterns: {enhancement_results.get('base_models_detected', 0)}")
            print(f"   Base models detected from files: {enhancement_results.get('base_models_detected_files', 0)}")
            print(f"   Models missing from filesystem: {enhancement_results.get('models_missing', 0)}")
            print(f"   Models updated in database: {enhancement_results.get('models_updated', 0)}")
            print(f"   Errors: {enhancement_results.get('errors', 0)}")
        
        # Step 3: Process orphaned media files
        if args.process_orphaned:
            print(f"\n🔄 Processing orphaned media files...")
            orphaned_files = scanner.get_orphaned_media_files()
            
            if orphaned_files:
                print(f"Found {len(orphaned_files)} orphaned media files")
                stats = scanner.process_orphaned_media(orphaned_files, dry_run=args.dry_run)
                
                print(f"✅ Orphaned media processing completed:")
                print(f"   Processed: {stats['processed']}")
                print(f"   Moved to model directories: {stats['moved_to_model']}")
                print(f"   Moved to duplicates: {stats['moved_to_duplicates']}")
                print(f"   Marked as ignored: {stats['ignored']}")
                print(f"   Errors: {stats['errors']}")
                
                total_stats['orphaned_processed'] = stats['processed']
            else:
                print("No orphaned media files found")
        
        # Step 4: Sort model files
        if args.sort_models:
            print(f"\n📁 Sorting model files into organized structure (batch size configured in config.ini)...")
            sort_results = scanner.sort_models_batch()
            
            print(f"\n✅ Model sorting results:")
            print(f"   Total models processed: {sort_results.get('total_models_processed', 0)}")
            print(f"   Models moved: {sort_results.get('models_moved', 0)}")
            print(f"   Associated files moved: {sort_results.get('associated_files_moved', 0)}")
            print(f"   Database batches committed: {sort_results.get('batches_committed', 0)}")
            print(f"   Errors: {sort_results.get('errors', 0)}")
        
        # Final summary
        print(f"\n📊 Final Summary:")
        print(f"   Total files scanned: {total_stats['total_files']}")
        print(f"   Models found: {total_stats['models_found']}")
        print(f"   Media files processed: {total_stats['media_processed']}")
        print(f"   Metadata extracted: {total_stats['metadata_extracted']}")
        print(f"   Orphaned media processed: {total_stats['orphaned_processed']}")
        
        if args.dry_run:
            print(f"\n💡 This was a DRY RUN - no files were actually moved.")
            print(f"   Remove --dry-run to perform actual file operations.")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        scanner.close()


if __name__ == "__main__":
    main()