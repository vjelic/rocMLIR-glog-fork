"""
This script compiles a given config in order to collect the following data
points for each config:
- Blocksize
- Gridsize
- vgpr
- sgpr
- LDS allocated
- Occupancy
- wf_per_wg
- mfma_wmma_instruction

The given config is expected to be a tsv with the following format:
|# arch| numCUs | testVector | perfConfig (exhaustive) |

Usage:
    python3 compileAndCollectTuningData.py --op <operation> <config.tsv>
"""

import argparse
import csv
import os
import re
import subprocess
import sys

from datetime import datetime
from testing_metrics import calculateGemmOccupancy, calculateAttentionOccupancy

# This script expects that ninja ci-performance-scripts has already been run
import perfRunner


# TODO use AmdArchDb.py (when it's implemented). 4 works for all current
# architectures, but this may not hold in the future.
numEUPerCU = 4

# Constants for individual field names
FIELD_ARCH = 'arch'
FIELD_NUM_CUS = 'numCUs'
FIELD_TEST_VECTOR = 'testVector'
FIELD_PERF_CONFIG = 'PerfConfig'
FIELD_BLOCKSIZE = 'blocksize'
FIELD_GRIDSIZE = 'gridsize'
FIELD_VGPR_COUNT = 'vgpr_count'
FIELD_VGPR_SPILLS = 'vgpr_spills'
FIELD_SGPR_COUNT = 'sgpr_count'
FIELD_SGPR_SPILLS = 'sgpr_spills'
FIELD_LDS_ALLOCATED = 'lds_allocated'
FIELD_OCCUPANCY = 'occupancy'
FIELD_WF_PER_WG = 'wf_per_wg'
FIELD_MFMA_WMMA_INSTRUCTION = 'mfma_wmma_instruction'

TSV_FIELDNAMES = [
    FIELD_ARCH,
    FIELD_NUM_CUS, 
    FIELD_TEST_VECTOR,
    FIELD_PERF_CONFIG,
    FIELD_BLOCKSIZE,
    FIELD_GRIDSIZE,
    FIELD_VGPR_COUNT,
    FIELD_VGPR_SPILLS,
    FIELD_SGPR_COUNT,
    FIELD_SGPR_SPILLS,
    FIELD_LDS_ALLOCATED,
    FIELD_OCCUPANCY,
    FIELD_WF_PER_WG,
    FIELD_MFMA_WMMA_INSTRUCTION
]

class TuningData:
    """Class to represent tuning data results."""
    
    def __init__(self):
        self.blocksize = None
        self.gridsize = None
        self.vgpr_count = None
        self.vgpr_spills = None
        self.sgpr_count = None
        self.sgpr_spills = None
        self.lds_allocated = None
        self.occupancy = None
        self.wf_per_wg = None
        self.mfma_wmma_instruction = None
    
    def to_dict(self):
        """Convert to dictionary format for tsv writing."""
        return self.__dict__

def get_perf_config(operation, test_vector, arch, num_cu):
    """
    Get the performance configuration for the given test vector, architecture,
    and number of compute units.
    
    Args:
        test_vector: The test vector string.
        arch: The architecture string.
        num_cu: The number of compute units.
    
    Returns:
        str: The performance configuration string.
    """
    conf_class = perfRunner.PerfConfiguration
    if (operation == 'attention'):
        conf_class = perfRunner.AttentionConfiguration.fromCommandLine(test_vector.split(sep=' '), arch, num_cu)
    elif (operation == 'gemm'):
        conf_class = perfRunner.GemmConfiguration.fromCommandLine(test_vector.split(sep=' '), arch, num_cu)
    elif (operation == 'conv'):
        conf_class = perfRunner.ConvConfiguration.fromCommandLine(test_vector.split(sep=' '), arch, num_cu)

    return conf_class

def compile_config(conf_class, operation, paths, timestamp):
    rocmlir_gen_options = conf_class.generateMlirDriverCommandLine("", None)

    # Build the rocmlir-gen command
    rocmlir_gen_cmd = [paths.mlir_paths.rocmlir_gen_path] + rocmlir_gen_options.split()

    # Build the rocmlir-driver command
    rocmlir_driver_cmd = [
        paths.mlir_paths.rocmlir_driver_path,
        "-c",
        "--debug-only=convert-rock-to-gpu,serialize-to-isa",
    ]

    commands = [rocmlir_gen_cmd, rocmlir_driver_cmd]
    out, err = perfRunner.runPipeline(commands)
    
    return err

def parse_mfma_wmma_instructions(content):
    """
    Parse MFMA and WMMA instructions from the debug output.
    
    Args:
        content: String content of the debug output file
        
    Returns:
        list: Unique list of MFMA/WMMA instruction names
    """
    # Pattern to match MFMA and WMMA instructions
    full_pattern = r'\b(v_(?:mfma|wmma)_[a-zA-Z0-9_]+)\b'
    full_matches = re.findall(full_pattern, content, re.IGNORECASE)
    
    # Remove duplicates and sort for consistent output
    unique_instructions = list(set(full_matches))
    
    # Assert that there is only one unique instruction
    size = len(unique_instructions)
    assert size <= 1, \
           f"Expected exactly one unique MFMA/WMMA instruction, found: {size}"
    
    return unique_instructions

def parse_results(debug_output):
    """
    This function parses the generated output file to gather the desired
    information.debug_output will contain all of the output from running
    rocmlir-driver (debug output and assembly output). It will be structured
    something like the following:
    """
    tuning_data = TuningData()

    # Look for blocksize
    blocksize_match = re.search(r'blockSize:\s*(\d+)', debug_output)
    if not blocksize_match:
        raise ValueError(f"Could not find blockSize in output")
    tuning_data.blocksize = int(blocksize_match.group(1))

    # Look for gridsize
    gridsize_match = re.search(r'gridSize:\s*(\d+)', debug_output)
    if not gridsize_match:
        raise ValueError(f"Could not find gridSize in output")
    tuning_data.gridsize = int(gridsize_match.group(1))

    # Look for waveSize
    wavesize_match = re.search(r'waveSize:\s*(\d+)', debug_output)
    if not wavesize_match:
        raise ValueError(f"Could not find waveSize in output")
    tuning_data.wf_per_wg = int(blocksize_match.group(1)) / int(wavesize_match.group(1))

    # Look for lds_allocated
    lds_match = re.search(r'ldsUsage:\s*(\d+)', debug_output)
    if not lds_match:
        raise ValueError(f"Could not find ldsUsage in output")
    tuning_data.lds_allocated = int(lds_match.group(1))

    # Look for SGPR count
    sgpr_match = re.search(r'\.sgpr_count:\s+(\d+)', debug_output)
    if not sgpr_match:
        raise ValueError(f"Could not find sgpr_count in output")
    tuning_data.sgpr_count = int(sgpr_match.group(1))
    
    # Look for VGPR count
    vgpr_match = re.search(r'\.vgpr_count:\s+(\d+)', debug_output)
    if not vgpr_match:
        raise ValueError(f"Could not find vgpr_count in output")
    tuning_data.vgpr_count = int(vgpr_match.group(1))
    
    # Look for SGPR spill count
    sgpr_spill_match = re.search(r'\.sgpr_spill_count:\s+(\d+)',
                                    debug_output)
    if not sgpr_spill_match:
        raise ValueError(f"Could not find sgpr_spill_count in output")
    tuning_data.sgpr_spills = int(sgpr_spill_match.group(1))
    
    # Look for VGPR spill count
    vgpr_spill_match = re.search(r'\.vgpr_spill_count:\s+(\d+)',
                                    debug_output)
    if not vgpr_spill_match:
        raise ValueError(f"Could not find vgpr_spill_count in output")
    tuning_data.vgpr_spills = int(vgpr_spill_match.group(1))

    mfma_wmma_instructions = parse_mfma_wmma_instructions(debug_output)
    if mfma_wmma_instructions:
        tuning_data.mfma_wmma_instruction = mfma_wmma_instructions[0]

    return tuning_data

def calculateNPerWave(n_per_wave, m_per_wave, n_per_block, m_per_block, arch):
    """
    Calculate the NPerWave value based on the given NPerWave value and the
    architecture that we are targeting
    """
    # Split at the first ':' if it exists
    if ':' in arch:
        arch = arch.split(':', 1)[0]

    # For CDNA architectures (gfx9xx) the n_per_wave value passed in is really
    # mnPerXdl, so we need to calculate the actual n_per_wave value 
    if arch.startswith('gfx9'):
        # This should always match with what the value for maxWavesPerWG is in
        # Rock.h
        max_waves_per_wg = 4
    
        m_waves = min(m_per_block / m_per_wave, max_waves_per_wg)
        n_waves = max_waves_per_wg / m_waves
        return max(n_per_block / n_waves, n_per_wave)

    # For RDNA architectures (gfx10xx, gfx11xx, gfx12xx) we can just use the
    # n_per_wave value as is
    return n_per_wave
    
def parse_perf_config(perf_config, num_cu, arch):
    """
    Parse the perfConfig string to extract tuning parameters.
    
    Format: attn:v1:MPerBlock,NPerBlock,KPerBlock,MPerWave,NPerWave,kPack,
            splitKFactor,forceUnroll,ThreadCopyMore
    
    Returns:
        dict: Dictionary containing parsed parameters

    TODO: The format of the perfConfig string is subject to changes in the
          future, so we should at a minimum be keeping this in sync with the
          c++ code, but we should als consider making bindings to the c++ code
          that can be called from here.
    """
    try:
        # Split by ':' to separate operation, version, and parameters
        parts = perf_config.split(':')
        if len(parts) < 2:
            raise ValueError(f"Invalid perfConfig format: {perf_config}")
        
        # The format is either going to have three parts or two parts. Make sure
        # to properly handle the `operation` case
        # - operation:version:parameters
        # - version:parameters
        version = None
        if len(parts) >= 3:
            # If there are three parts, then we assume the first part denoting
            # the operation is going to be equal to `attn`
            assert(parts[0] == 'attn')
            params_str = parts[2]
            version = parts[1]
        else:
            # parameters are after the first ':'
            params_str = parts[1]
            version = parts[0]
        
        # Split parameters by comma
        params = params_str.split(',')
        if ((version == "v1") and not (len(params) == 8)) \
           or ((version == "v2") and not (len(params) == 9)) \
           or ((version == "v3") and not (len(params) == 11)):
            raise ValueError(f"Insufficient parameters in perfConfig")
        
        # Parse the required parameters. Note that kpack does not exist in v1,
        # so we set it to 1
        parsed_params = {
            'MPerBlock': int(params[0]),
            'NPerBlock': int(params[1]),
            'KPerBlock': int(params[2]),
            'MPerWave': int(params[3]),
            'NPerWave': calculateNPerWave(int(params[4]), int(params[3]),
                                          int(params[1]),
                                          int(params[0]), arch),
            'kPack': 1 if (version == "v1") else int(params[5]),
            'splitKFactor': int(params[6])
        }
        
        # Calculate M*N PerWave
        parsed_params['MNPerWave'] = parsed_params['MPerWave'] * \
                                     parsed_params['NPerWave']

        # Calculate minNumWaves based on numCUs and numEUPerCU
        parsed_params['minNumWaves'] = int(num_cu) * numEUPerCU
        
        return parsed_params
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing perfConfig '{perf_config}': {e}")
        return None
    
def extract_MNG_from_config(conf_class, operation):
    """
    Extract M, N, and G values from the testVector based on the operation type.
    
    Args:
        conf_class: Configuration class instance of specified operation
        operation: Operation type (e.g., 'attention', 'gemm', 'conv2d')
    
    Returns:
        tuple: (M, N, G) values based on operation type
    """
    try:
        if operation.lower() in ['attention', 'attn']:
            # For attention ops: M = seq_len_q, N = seq_len_k, G = g * num_heads_q
            M = conf_class.seq_len_q
            N = conf_class.seq_len_k
            G = conf_class.g * conf_class.num_heads_q
            
        elif operation.lower() in ['gemm']:
            # For GEMM ops: M = m, N = n, G = g
            M = conf_class.m
            N = conf_class.n
            G = conf_class.g
            
        elif operation.lower() in ['conv', 'convfp16', 'convbfp16', 'convint8',
                                   'convfp8']:
            # For conv ops: M = k, N = batch_size * output_height * output_width,
            # G = g
            assert conf_class.direction == 'fwd', \
           "Only forward convolution (-F=1) is supported"
            G = conf_class.g
            M = conf_class.k
            N = conf_class.n * conf_class.ho * conf_class.wo
            
        else:
            print(f"Warning: Unknown operation type '{operation}'")
            return None, None, None
            
    except (ValueError, TypeError) as e:
        print(f"Warning: Error parsing M, N, G values from testVector: {e}")
        print(f"testVector: {test_vector}")
        print(f"Parsed args: {arg_dict}")
        return None, None, None
    
    return M, N, G

def gatherOccupancyParameters(config, perf_config, conf_class, operation):
    '''
    This function gathers all of the parameters that are needed to calculate
    the theoretical occupancy
    '''
    num_cu = config[1]
    test_vector = config[2]
    parsed_params = parse_perf_config(perf_config, num_cu, config[0])
    
    if parsed_params is None:
        return [None] * 8  # Return None values if parsing fails
    
    # Extract the required parameters for occupancy calculation
    [M, N, G] = extract_MNG_from_config(conf_class, operation)
    
    MPerBlock = int(parsed_params['MPerBlock'])
    NPerBlock = int(parsed_params['NPerBlock'])
    MNPerWave = int(parsed_params['MNPerWave'])
    minNumWaves = int(parsed_params['minNumWaves'])
    splitKFactor = int(parsed_params['splitKFactor'])

    return [M, N, G, MPerBlock, NPerBlock, MNPerWave, minNumWaves, splitKFactor]

def compile_and_collect_data(config, operation, binaries):
    """
    Compile and collect the resulting data points that we are interested in
    """
    # Get current timestamp in a filesystem-friendly format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a performance configuration class instance
    arch = config[0].split(':')[0]
    num_cu = config[1]
    test_vector = config[2]
    perf_config = config[3]
    tflops = config[4]
    conf_class = get_perf_config(operation, test_vector, arch, num_cu)
    conf_class.setPerfConfig(perf_config)

    # Compile the config
    debug_output = compile_config(conf_class, operation, binaries,
                                  timestamp)
    if isinstance(debug_output, bytes):
        debug_output = debug_output.decode('utf-8')

    # If the debug output is empty, then this means that the compilation
    # pipeline failed. We expect this to happe for some of the invalid configs
    if not debug_output:
        print(f"Warning: Compilation failed for config {config}. "
              "Skipping calculations.")
        return None

    # Parse the results from the compiled config
    results = parse_results(debug_output)

    # TODO: Convert the TFLOPs value to seconds

    # Calculate occupancy using the method in testing_metrics.py
    [M, N, G, MPerBlock, NPerBlock,
        MNPerWave, minNumWaves, splitKFactor] = \
                                gatherOccupancyParameters(config,
                                                          perf_config,
                                                          conf_class,
                                                          operation)
    # If any of the parameters are None, we cannot calculate occupancy
    if None in [M, N, G, MPerBlock, NPerBlock,
                MNPerWave, minNumWaves, splitKFactor]:
        print("Warning: Could not gather all parameters for occupancy "
              "calculation for config. Skipping occupancy calculation.")
        results.occupancy = None
    elif operation.lower() == 'attention':
        results.occupancy = calculateAttentionOccupancy(N, G, MPerBlock,
                                                        NPerBlock,
                                                        MNPerWave, minNumWaves)
    else :
        results.occupancy = calculateGemmOccupancy(M, N, G, MPerBlock, NPerBlock,
                                               MNPerWave, minNumWaves,
                                               splitKFactor)

    return results

def write_results_to_tsv(results, configs):
    """
    Write the collected tuning data results to a tsv file.
    
    Args:
        results: List of tuning data dictionaries
        configs: List of original configuration dictionaries
    """
    if not results:
        print("No results to write")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tuning_results_{timestamp}.tsv"
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=TSV_FIELDNAMES)
            
            # Write the header
            writer.writeheader()
            
            # Write each result row
            for (config, result) in zip(configs, results):
                arch, num_cu, test_vector = config
                row = {
                    FIELD_ARCH: arch,
                    FIELD_NUM_CUS: num_cu,
                    FIELD_TEST_VECTOR: test_vector,
                    FIELD_PERF_CONFIG: configs[config],
                }

                data_fieldnames = TSV_FIELDNAMES[4:]
                if result is None:
                    row.update({field: None for field in data_fieldnames})
                else:
                    result_dict = result.to_dict()
                    row.update({field: result_dict.get(field, '') for field in data_fieldnames})
                writer.writerow(row)
        
        print(f"\nResults written to {output_file}")
        
    except Exception as e:
        print(f"\nError writing results to tsv: {e}")
        sys.exit(1)

def print_progress(current, total):
    """Print a progress bar to stdout."""
    prefix = "Processing Configs"
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percent:.1f}%)', end='',
          flush=True)
    if current == total:
        print()  # New line when complete

def main():
    """Main function to process configurations and collect tuning data."""
    parser = argparse.ArgumentParser(
        description="Compile configurations and collect tuning data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--op', "--operation", required=True,
                        help='Operation to perform (e.g., "compile")',
                        choices=['conv', 'gemm', 'attention'])
    parser.add_argument('config_tsv', help='Path to the tuning database file')
    
    args = parser.parse_args()

    # Get the paths to the rocmlir binaries
    build_bin_dir = os.path.dirname(os.path.abspath(__file__))
    rocmlir_root = os.path.dirname(build_bin_dir)
    paths = perfRunner.create_paths(None, rocmlir_root)

    # Check if the input config tsv file exists
    if not os.path.exists(args.config_tsv):
        print(f"Error: The specified config tsv file cannot be found.")
        return 1

    # Parse the configuration file
    configs = perfRunner.read_debug_db(args.config_tsv)

    print(f"Found {len(configs)} configurations to process")
    
    # Process each configuration
    results = []
    total_configs = len(configs)
    for i, config in enumerate(configs):
        print_progress(i, total_configs)
        metrics = compile_and_collect_data(config, args.op, paths)
        results.append(metrics)

    # Write the results to a final tsv file
    write_results_to_tsv(results, configs)

    # If we have reached this point without crashing, it means that we have had
    # a successful run and we can return 0.
    return 0

if __name__ == "__main__":
    sys.exit(main())
