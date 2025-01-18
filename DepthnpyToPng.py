#!/usr/bin/env python3
import numpy as np
import cv2
import argparse
import os
from pathlib import Path

def depth_to_png(depth_array, output_path, normalize=True, depth_scale=1000):
    """
    Convert a depth image from numpy array to PNG format
    
    Args:
        depth_array: Input depth array (from .npy file)
        output_path: Path to save the PNG file
        normalize: If True, normalize values to 0-255 range. If False, scale values by depth_scale
        depth_scale: Scale factor for depth values (default: 1000 for millimeter precision)
    """
    if depth_array.dtype != np.float32:
        depth_array = depth_array.astype(np.float32)

    if normalize:
        # Normalize to 0-255 range
        depth_min = np.min(depth_array[depth_array > 0])  # Ignore zero values
        depth_max = np.max(depth_array)

        # Create normalized image
        depth_normalized = np.zeros_like(depth_array)
        non_zero_mask = depth_array > 0
        depth_normalized[non_zero_mask] = ((depth_array[non_zero_mask] - depth_min) /
                                           (depth_max - depth_min) * 255)
        depth_image = depth_normalized.astype(np.uint8)

    else:
        # Scale depth values directly
        depth_image = (depth_array * depth_scale).astype(np.uint16)

    # Apply colormap for better visualization
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=1),
                                       cv2.COLORMAP_JET)

    # Save both grayscale and colored versions
    cv2.imwrite(str(output_path), depth_image)
    colored_path = output_path.parent / (output_path.stem + '_colored.png')
    cv2.imwrite(str(colored_path), depth_colormap)

    return depth_image

def convert_npy_to_png(npy_path, png_path, normalize=True, depth_scale=1000):
    """
    Load a .npy depth file and convert it to PNG
    
    Args:
        npy_path: Path to input .npy file
        png_path: Path to save output PNG file
        normalize: Whether to normalize the depth values
        depth_scale: Scale factor for depth values
    """
    try:
        depth_array = np.load(npy_path)
        depth_image = depth_to_png(depth_array, png_path, normalize, depth_scale)
        print(f"Successfully converted {npy_path} to {png_path}")
        return True
    except Exception as e:
        print(f"Error converting {npy_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert depth images from NPY to PNG format')
    parser.add_argument('input', type=str, help='Input .npy file or directory')
    parser.add_argument('--output', type=str, help='Output directory (optional)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable normalization of depth values')
    parser.add_argument('--depth-scale', type=float, default=1000,
                        help='Scale factor for depth values (default: 1000 for millimeters)')

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return

    # Handle output directory
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = input_path if input_path.is_dir() else input_path.parent
    output_base.mkdir(parents=True, exist_ok=True)

    # Process single file
    if input_path.is_file():
        if input_path.suffix != '.npy':
            print(f"Error: Input file must be .npy format: {input_path}")
            return

        output_path = output_base / (input_path.stem + '.png')
        convert_npy_to_png(input_path, output_path,
                           normalize=not args.no_normalize,
                           depth_scale=args.depth_scale)

    # Process directory
    else:
        npy_files = list(input_path.glob('*.npy'))
        if not npy_files:
            print(f"No .npy files found in {input_path}")
            return

        print(f"Found {len(npy_files)} .npy files to convert")
        success_count = 0

        for npy_file in npy_files:
            output_path = output_base / (npy_file.stem + '.png')
            if convert_npy_to_png(npy_file, output_path,
                                  normalize=not args.no_normalize,
                                  depth_scale=args.depth_scale):
                success_count += 1

        print(f"\nConversion complete: {success_count}/{len(npy_files)} files converted successfully")

if __name__ == '__main__':
    main()