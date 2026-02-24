"""
===============================================================================
1_vlm_demo.py - Vision Language Model Demo for 3D Voxel Generation
===============================================================================

This script uses a fine-tuned Qwen2.5-VL model to analyze images and generate
3D voxel representations of objects and their parts.

Pipeline Overview:
    1. Load an image from the demo folder
    2. Send image + prompt to the VLM to get basic object info (parts, materials)
    3. For each detected part, generate voxel coordinates in a 32x32x32 grid
    4. Save voxel data as numpy arrays and optionally as PLY point clouds

Key Concepts:
    - Voxel Grid: A 32x32x32 3D grid where each cell can be occupied or empty
    - Voxel Encoding: 3D coordinates (x,y,z) are encoded into a single integer
      using bit shifting: index = (x << 10) | (y << 5) | z
    - Run-Length Encoding: Consecutive voxel indices are merged (e.g., "199-216")

Author: PhysX-Anything Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Machine Learning & Vision
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Standard Library
import base64
import os
import argparse

# Data Processing
import numpy as np
from PIL import Image
import trimesh

# Image Processing
from rembg import remove

# Debugging (can be removed in production)
import ipdb


# =============================================================================
# VOXEL ENCODING/DECODING UTILITIES
# =============================================================================

def voxel_encode(voxels: np.ndarray, size: int = 32) -> np.ndarray:
    """
    Encode 3D voxel coordinates into single integers using bit-packing.
    
    This function converts (x, y, z) coordinates into a single integer index
    using bit shifting. Each coordinate uses 5 bits (supports values 0-31).
    
    Formula: index = (x << 10) | (y << 5) | z
    
    Args:
        voxels (np.ndarray): Array of shape (N, 3) containing voxel coordinates
        size (int): Grid size (must be 32 for 5-bit encoding)
    
    Returns:
        np.ndarray: Array of encoded integer indices
    
    Example:
        >>> coords = np.array([[0, 0, 0], [1, 2, 3]])
        >>> voxel_encode(coords)
        array([0, 1091])
    """
    voxels = np.asarray(voxels, dtype=np.int64)
    
    # Validate input shape
    assert voxels.ndim == 2 and voxels.shape[1] == 3, "voxels shape should be (N, 3)"
    assert size == 32, "Grid size must be 32 (2^5) for 5-bit encoding"
    
    # Validate coordinate ranges
    if (voxels < 0).any() or (voxels >= size).any():
        raise ValueError("All coordinates must be in range [0, 32)")
    
    # Extract individual coordinates
    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    
    # Bit-pack into single integer: x uses bits 10-14, y uses bits 5-9, z uses bits 0-4
    return (x << 10) | (y << 5) | z


def voxel_decode(indices: np.ndarray, size: int = 32) -> np.ndarray:
    """
    Decode integer indices back into 3D voxel coordinates.
    
    This reverses the voxel_encode operation, extracting (x, y, z) coordinates
    from bit-packed integers using bit masking and shifting.
    
    Args:
        indices (np.ndarray): Array of encoded voxel indices
        size (int): Grid size (must be 32)
    
    Returns:
        np.ndarray: Array of shape (N, 3) containing decoded coordinates
    
    Example:
        >>> indices = np.array([0, 1091])
        >>> voxel_decode(indices)
        array([[0, 0, 0], [1, 2, 3]])
    """
    indices = np.asarray(indices, dtype=np.int64).ravel()
    
    assert size == 32, "Grid size must be 32 (2^5) for 5-bit decoding"
    
    # Clamp out-of-range indices and warn
    if (indices < 0).any() or (indices >= size**3).any():
        indices = indices.clip(0, size**3 - 1)
        print("Warning: Some indices were out of range [0, 32768) and have been clamped.")
    
    # Extract coordinates using bit masking (31 = 0b11111, a 5-bit mask)
    x = (indices >> 10) & 31  # Extract bits 10-14
    y = (indices >> 5) & 31   # Extract bits 5-9
    z = indices & 31          # Extract bits 0-4
    
    return np.stack([x, y, z], axis=1)


# =============================================================================
# STRING CONVERSION UTILITIES (for VLM communication)
# =============================================================================

def ints_to_space_separated_str(arr: np.ndarray) -> str:
    """
    Convert an array of integers to a space-separated string.
    
    Args:
        arr (np.ndarray): Array of integers
    
    Returns:
        str: Space-separated string representation
    
    Example:
        >>> ints_to_space_separated_str(np.array([1, 2, 3]))
        '1 2 3'
    """
    arr = np.asarray(arr, dtype=np.int64).ravel()
    return " ".join(map(str, arr))


def merge_adjacent_to_dash(s: str) -> str:
    """
    Compress a sequence of numbers by merging consecutive runs into ranges.
    
    This function takes a space-separated string of numbers and converts
    consecutive sequences into dash-separated ranges for more compact output.
    
    Args:
        s (str): Space-separated string of integers (e.g., "1 2 3 5 6 7 10")
    
    Returns:
        str: Compressed string with ranges (e.g., "1-3 5-7 10")
    
    Example:
        >>> merge_adjacent_to_dash("199 200 201 202 230 231")
        '199-202 230-231'
    """
    if not s.strip():
        return ""
    
    # Parse, sort, and deduplicate numbers
    nums = list(map(int, s.split()))
    nums = sorted(set(nums))
    
    # Build ranges from consecutive sequences
    result = []
    start = prev = nums[0]
    
    for n in nums[1:]:
        if n == prev + 1:
            # Continue current range
            prev = n
        else:
            # End current range and start new one
            result.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = n
    
    # Don't forget the last range
    result.append(f"{start}-{prev}" if start != prev else f"{start}")
    
    return " ".join(result)


def dash_str_to_ints(s: str) -> np.ndarray:
    """
    Expand a compressed dash-notation string back into individual integers.
    
    This reverses the merge_adjacent_to_dash operation, expanding ranges
    like "199-202" back into individual numbers [199, 200, 201, 202].
    
    Args:
        s (str): Compressed string with ranges (e.g., "199-202 230-231")
    
    Returns:
        np.ndarray: Sorted, deduplicated array of integers
    
    Example:
        >>> dash_str_to_ints("1-3 5-7 10")
        array([1, 2, 3, 5, 6, 7, 10])
    """
    if not s.strip():
        return np.array([], dtype=np.int64)
    
    out = []
    for token in s.split():
        if "-" in token:
            # Expand range notation
            a, b = map(int, token.split("-"))
            if a > b:
                a, b = b, a  # Handle reversed ranges
            out.extend(range(a, b + 1))
        else:
            # Single number
            out.append(int(token))
    
    return np.array(sorted(set(out)), dtype=np.int64)


# =============================================================================
# CONVERSATION MANAGEMENT FOR VLM
# =============================================================================

def addmessage(message, before, after):
    """
    Append a Q&A pair to the conversation history.
    
    This function simulates a conversation turn by adding the model's response
    (before) and the user's follow-up question (after) to the message history.
    
    Args:
        message (list): Current conversation history
        before (str): Assistant's response text
        after (str): User's next question text
    
    Returns:
        list: Updated conversation history with new Q&A pair
    """
    # Create assistant response
    answer = {
        'role': 'assistant',
        'content': [{"type": "text", "text": before}]
    }
    
    # Create user follow-up question
    question = {
        'role': 'user',
        'content': [{"type": "text", "text": after}]
    }
    
    # Append to conversation (copy to avoid modifying original)
    newmessage = message.copy()
    newmessage.append(answer)
    newmessage.append(question)
    
    return newmessage


# =============================================================================
# MODEL INFERENCE
# =============================================================================

def generate_save(model, messages, save_dir, save_name='test', save=True):
    """
    Generate a response from the VLM and optionally save it to a file.
    
    This function processes the conversation, runs inference through the model,
    and extracts the generated text response.
    
    Args:
        model: The loaded VLM model
        messages (list): Conversation history in OpenAI-style format
        save_dir (str): Directory to save the output
        save_name (str): Base filename for the output (without extension)
        save (bool): Whether to save the output to a file
    
    Returns:
        str: The model's generated text response
    """
    # Prepare input using the chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process any images/videos in the messages
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Tokenize and prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate response (deterministic with temperature=0)
    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=0,
        max_length=32768  # Allow long outputs for voxel coordinates
    )
    
    # Extract only the newly generated tokens (exclude input prompt)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode tokens to text
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    # Save output if requested
    if save:
        output_path = os.path.join(save_dir, save_name + '.txt')
        with open(output_path, 'w') as file:
            file.write(output_text[0])
    
    return output_text[0]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Parse Command Line Arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="VLM Demo: Generate 3D voxel representations from images"
    )
    parser.add_argument(
        "--demo_path", type=str, default='./demo',
        help="Path to input images directory"
    )
    parser.add_argument(
        "--save_part_ply", type=bool, default=True,
        help="Whether to save individual part point clouds as PLY files"
    )
    parser.add_argument(
        "--remove_bg", type=bool, default=False,
        help="Whether to remove background from input images"
    )
    parser.add_argument(
        "--ckpt", type=str, default='./pretrain/vlm',
        help="Path to the fine-tuned VLM checkpoint"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup: Load Input Files and Model
    # -------------------------------------------------------------------------
    basepath = args.demo_path
    namelist = os.listdir(basepath)
    
    print(f"Found {len(namelist)} images in {basepath}")
    
    # Load the Vision-Language Model with optimizations
    print("Loading VLM model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,          # Use bfloat16 for memory efficiency
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 for speed
        device_map="auto",                    # Automatically distribute across GPUs
    )
    
    # Configure image processor with resolution limits
    # These settings balance detail vs. memory usage
    min_pixels = 65536   # 256x256 minimum
    max_pixels = 262144  # 512x512 maximum
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.size["shortest_edge"] = min_pixels
    processor.image_processor.size["longest_edge"] = max_pixels

    # -------------------------------------------------------------------------
    # Process Each Image
    # -------------------------------------------------------------------------
    for name in namelist:
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print('='*60)
        
        # Create output directory for this image
        save_dir = os.path.join('test_demo', name[:-4])  # Remove file extension
        os.makedirs(save_dir, exist_ok=True)
        
        image_path = os.path.join(basepath, name)
        
        # Load the prompt template for object analysis
        with open('./dataset/overall_prompt.txt', "r", encoding="utf-8") as f:
            basicqu = f.read()
        
        # Load and resize input image
        input_image = Image.open(image_path)
        im_resized = input_image.resize((512, 512), Image.LANCZOS)
        
        # Optionally remove background for cleaner analysis
        if args.remove_bg:
            im_resized = remove(im_resized)
        
        # ---------------------------------------------------------------------
        # Step 1: Get Basic Object Information
        # ---------------------------------------------------------------------
        # Initial message with image and analysis prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": im_resized.convert("RGB"),
                    },
                    {"type": "text", "text": basicqu},
                ],
            }
        ]
        
        # Get basic object info (name, category, parts, materials, etc.)
        print("Step 1: Analyzing object structure...")
        basicoutput = generate_save(model, messages, save_dir, 'basic_info')
        
        # Count how many parts were detected (look for l_0, l_1, l_2, etc.)
        num_parts = 0
        while f'l_{num_parts}' in basicoutput:
            num_parts += 1
        print(f"Detected {num_parts} parts")
        
        # ---------------------------------------------------------------------
        # Step 2: Generate Voxel Coordinates for Each Part
        # ---------------------------------------------------------------------
        allcoord = []  # Collect all part coordinates
        
        for part in range(num_parts):
            print(f"Step 2.{part}: Generating voxels for part l_{part}...")
            
            # Construct prompt for voxel generation
            # The model outputs voxel indices in compressed dash notation
            question = (
                f"Based on the structured description of l_{part}, "
                f"generate its 3D voxel grid in the following format "
                f"(voxel grid=32, use numbers from 0 to 32767, "
                f"merge maximal consecutive runs: 199...216 -> 199-216): "
                f"184 198 199-216 230-237..."
            )
            
            # Add previous response and new question to conversation
            messages1 = addmessage(messages, basicoutput, question)
            
            # Generate voxel coordinates
            output1 = generate_save(model, messages1, save_dir, f'coord_{part}', save=True)
            print(f"  Conversation length: {len(messages1)} messages")
            
            # Decode the compressed voxel indices back to 3D coordinates
            idx_back = dash_str_to_ints(output1)
            voxels_back = voxel_decode(idx_back)
            
            print(f"  Generated {len(voxels_back)} voxels for part {part}")
            
            # Save individual part data
            allcoord.append(voxels_back)
            np.save(os.path.join(save_dir, f'ind_{part}.npy'), voxels_back)
            
            # Optionally save as PLY point cloud for visualization
            if args.save_part_ply:
                partply = trimesh.points.PointCloud(voxels_back)
                partply.export(os.path.join(save_dir, f'ind_{part}.ply'))
        
        # ---------------------------------------------------------------------
        # Step 3: Save Combined Voxel Data
        # ---------------------------------------------------------------------
        if allcoord:
            combined_voxels = np.concatenate(allcoord)
            np.save(os.path.join(save_dir, 'allind.npy'), combined_voxels)
            print(f"Saved combined voxels: {len(combined_voxels)} total voxels")
        
        print(f"Completed processing: {name}")


