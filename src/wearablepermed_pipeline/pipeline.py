import os
import sys
import random
import argparse
import logging
import subprocess
import pandas as pd
from enum import Enum
from os import walk, path
from datetime import datetime
from pathlib import Path

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

_DEF_ARG_WINDOW_OVERLAPPING_PERCENT = None

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'

class ML_Sensor(Enum):
    PI = 'thigh'
    M = 'wrist'
    C = 'hip'

def parse_steps(s: str):
    # Split input by commas, strip spaces, and convert to int
    try:
        steps = [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("All steps must be integers.")

    # Validate: must not be empty
    if not steps:
        raise argparse.ArgumentTypeError("At least one step must be provided.")

    # Validate: all must be 1–5
    valid_steps = {1, 2, 3, 4, 5, 6}
    invalid = [x for x in steps if x not in valid_steps]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid steps: {invalid}. Only 1, 2, 3, 4, 5 or 6 are allowed."
        )

    # Validate: ascending order
    if steps != sorted(steps):
        raise argparse.ArgumentTypeError("Steps must be in ascending order (e.g. 1,2,3).")
    
    return steps
    
def parse_args(args):
    """Parse command line parameters

    Args:hip
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="WareablePerMed Pipeline")

    parser.add_argument(
        "-execute-steps",
        "--execute-steps",
        dest="execute_steps", 
        #type=lambda s: [int(x) for x in s.split(",")],                
        type=parse_steps,
        required=True,
        help="Choose the steps to be executed."
    )

    parser.add_argument(
        "-dataset-folder",
        "--dataset-folder",
        dest="dataset_folder",
        required=True,
        help="Choose the dataset root folder."
    )

    parser.add_argument(
        "-participants-missing-file",
        "--participants-missing-file",
        dest="participants_missing_file",        
        type=argparse.FileType("r"),
        required=True,
        help="Choose the missing participants csv file."
    )

    parser.add_argument(
        '-crop-columns', 
        '--crop-columns', 
        dest="crop_columns",
        help='Columns to select from arrays. Format: "start:end" or "col1,col2,col3". Default: "1:7"',
        default=slice(1, 7),
        required=True
    )
    
    parser.add_argument(
        '-window-size', 
        '--window-size',
        dest="window_size",
        help='Window size in number of samples',        
        required=True,
    )
    
    parser.add_argument(
        "-window-overlapping-percent",
        "--window-overlapping-percent",
        dest="window_overlapping_percent", 
        help="Window Overlapping percent."
    )  

    parser.add_argument(
        "-include-not-estructure-data",
        "--include-not-estructure-data",
        dest="include_not_estructure_data",
        action='store_true',
        help="Include estructure data."
    )     

    parser.add_argument(
        "-ml-models",
        "--ml-models",
        dest="ml_models",        
        required=True,
        help=f"Available ML models: {[c.value for c in ML_Model]}."
    )

    parser.add_argument(
        "-ml-sensors",
        "--ml-sensors",
        dest="ml_sensors",        
        required=True,
        help=f"Available ML sensors: {[c.value for c in ML_Sensor]}."
    )   

    parser.add_argument(
        "-output-case-folder",
        "--output-case-folder",
        dest="output_case_folder",
        required=True,
        help="output folder for all Cases."
    )

    parser.add_argument(
        "-case-id",
        "--case-id",
        dest="case_id",
        required=True,
        help="Case unique identifier."
    ) 

    parser.add_argument(
        "-participants",
        "--participants",
        dest="participants",
        type=str,
        help="select subspace-separated participants IDs"
    )

    parser.add_argument(
        "-desync-include-only-not-visual-participants",
        "--desync-include-only-not-visual-participants",
        dest="desync_include_only_not_visual_participants",
        action='store_true',
        help="Include all participants with and without close IMU datetime."
    ) 

    parser.add_argument(
        "-desync-participant-percent",
        "--desync-participant-percent",           
        dest="desync_participant_percent",
        type=int,
        help='Participant percent desynchronize'
    )

    parser.add_argument(
        "-desync-segment-body",
        "--desync-segment-body",           
        dest="desync_segment_body",
        type=str,
        choices=['PI', 'M', 'C'],
        help='Segment Body to be desynchronize'
    )

    parser.add_argument(
        "-desync-seconds",
        "--desync-seconds",           
        dest="desync_seconds",
        type=int,
        help='Seconds to be desynchronize'
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO.",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def execute_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read both streams
    stdout_lines = []
    stderr_lines = []
    
    for line in process.stdout:
        line = line.strip()
        stdout_lines.append(line)
        _logger.info("Processing " + line)

    for line in process.stderr:
        line = line.strip()
        stderr_lines.append(line)
        _logger.error("Error output " + line)

    process.wait()

    # Build full outputs
    stdout_text = "\n".join(stdout_lines)
    stderr_text = "\n".join(stderr_lines)

    # Check exit code
    if process.returncode != 0:
        error_msg = (
            f"Command failed with exit code {process.returncode}\n"
            f"Command: {' '.join(command)}\n"
            f"STDERR:\n{stderr_text}\n"
        )

        # Log the error
        _logger.error(error_msg)

        raise subprocess.CalledProcessError(
            returncode=process.returncode,
            cmd=command,
            output=error_msg
        )
    
def STEP01(args, participant_id):
    # set participant folder
    participant_folder = Path(path.join(args.dataset_folder, participant_id))

    # get target files and process each one: participant binary datasets
    files = list(participant_folder.glob("*.BIN"))

    for file in files:
        _logger.info("Processing file: " + str(file))  

        cmd = [
            "sensor_bin_to_csv",
            "--bin-file", str(file),
            "--csv-file", path.join(str(file.parent), file.stem + ".csv")
        ]

        execute_command(cmd)

def STEP02(args, df, participant_id, desync_body_segment=None, desync_seconds=None):
    # set participant folder
    participant_folder = Path(path.join(args.dataset_folder, participant_id))
    
    # get target files and process each one: participant csv datasets
    files = list(participant_folder.glob("*.csv"))

    # create participant activity excel file
    activity_file = path.join(str(participant_folder), participant_id + "_RegistroActividades.xlsx")
        
    for file in files:
        _logger.info("Processing file: " + str(file))  

        # get segment body
        body_segment = file.stem.split("_")[2].split(".")[0]

        # check if participant has missing data
        row_missing_data = df[(df["Participant"] == participant_id) & (df["Acceloremeters"] == body_segment)]

        if row_missing_data.empty:
            if body_segment == desync_body_segment:
                cmd = [
                    "csv_to_segmented_activity_desync",
                    "--csv-file", str(file),
                    "--excel-activity-log", activity_file,
                    "--body-segment", body_segment,
                    "--desync-seconds", str(desync_seconds),
                    "--output", path.join(str(participant_folder), file.stem + "_seg" + ".npz")
                ]
            else:
                cmd = [
                    "csv_to_segmented_activity",
                    "--csv-file", str(file),
                    "--excel-activity-log", activity_file,
                    "--body-segment", body_segment,
                    "--output", path.join(str(participant_folder), file.stem + "_seg" + ".npz")
                ]

            execute_command(cmd)
        else: 
            if args.desync_include_only_not_visual_participants is False:
                if body_segment == desync_body_segment:
                    cmd = [
                        "csv_to_segmented_activity_desync",
                        "--csv-file", str(file),
                        "--excel-activity-log", activity_file,
                        "--body-segment", body_segment,
                        "--desync-seconds", str(desync_seconds),
                        "--sample-init", str(row_missing_data.iloc[0]["Sample Numbers"]),
                        "--start-time", str(row_missing_data.iloc[0]["Excel Hour"]),
                        "--output", path.join(str(participant_folder), file.stem + "_seg" + ".npz"),
                    ]
                else:
                    cmd = [
                        "csv_to_segmented_activity",
                        "--csv-file", str(file),
                        "--excel-activity-log", activity_file,
                        "--body-segment", body_segment,
                        "--sample-init", str(row_missing_data.iloc[0]["Sample Numbers"]),
                        "--start-time", str(row_missing_data.iloc[0]["Excel Hour"]),
                        "--output", path.join(str(participant_folder), file.stem + "_seg" + ".npz"),
                    ]

                execute_command(cmd)    

def STEP03(args, participant_id):
    # set participant folder
    participant_folder = Path(path.join(args.dataset_folder, participant_id))

    # get target files and process each one: segmentation datasets
    files = list(participant_folder.glob("*_seg.npz"))

    for file in files:
        _logger.info("Processing file: " + str(file))
        
        cmd = [
            "segmented_activity_to_stack",
            "--npz-file", str(file),
            "--crop-columns", args.crop_columns,
            "--window-size", args.window_size,
            "--window-overlapping-percent", args.window_overlapping_percent,
            "--output", path.join(str(file.parent), file.stem.replace("_seg", "_tot") + ".npz")
        ]

        if args.include_not_estructure_data == True:
            cmd.append('--include-not-estructure-data')

        execute_command(cmd)        

def STEP04(args, participant_id):
    # set participant folder
    participant_folder = Path(path.join(args.dataset_folder, participant_id))

    # get target files and process each one: convolution datasets
    files = list(participant_folder.glob("*_tot.npz"))

    for file in files:
        _logger.info("Processing file: " + str(file))
        
        cmd = [
            "stack_to_features",
            "--stack-file", str(file),
            "--output", path.join(str(file.parent), file.stem.replace("_tot", "_tot_features") + ".npz")
        ]

        execute_command(cmd)   

def STEP05(args, participant_id):
    cmd = [
        "aggregate_windows_features",
        "--dataset-folder", path.join(args.dataset_folder, participant_id),
        "--ml-models", args.ml_models,
        "--ml-sensors", args.ml_sensors,           
        "--output-folder", path.join(args.dataset_folder, participant_id)
    ]

    execute_command(cmd)  

def STEP06(args):    
    cmd = [
        "model_aggregation",
        "--dataset-folder", args.dataset_folder,
        "--output-folder", args.output_case_folder,
        "--case-id", args.case_id
    ]

    execute_command(cmd) 

def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Pipeline starts here at " + str(datetime.now()))

    # Load the CSV file with missing datetime participants
    df = pd.read_csv(
        args.participants_missing_file,
        delimiter=";",     # or ';' or '\t' if needed
        encoding="utf-8",  # handle special characters
    )

    df_error = pd.DataFrame(columns=["participant_id", "step", "command"])

    # Start pre training pipelines for all participants 
    _logger.info("Start pre training pipelines for all participants")

    # Open the file for writing
    with open("error_log.csv", mode="w", newline="") as error_file:
        # get a random sample of participants to be desynchronized in the segmentation step to study influence in metrics
        desync_participant_ids = None
        if (args.desync_participant_percent is not None):
            n_total = len(os.listdir(args.dataset_folder))
            n_select = int(n_total * args.desync_participant_percent / 100)

            desync_participant_ids = sorted(random.sample(os.listdir(args.dataset_folder), n_select))

        for dataset_folder_path, participant_ids, filenames in walk(args.dataset_folder):
            # Only select the participants selected from params if is defined
            if args.participants is not None:
                participant_ids[:] = [d for d in participant_ids if d in args.participants.split()]

            participant_ids.sort()
            filenames.sort()

            # Only process one level of subfolders
            if dataset_folder_path == args.dataset_folder:
                # Execute the pipeline for each participant
                for participant_id in participant_ids:
                    _logger.info(f"Execute the training participant pipeline for: {participant_id}")

                    # Define STEP01: convert bin to csv files for each participant id
                    if 1 in args.execute_steps:
                        _logger.info(f"STEP01: convert bin to csv files pipeline step for: {participant_id}")  

                        try:
                            STEP01(args, participant_id)    
                        except subprocess.CalledProcessError as e:
                            df_error.loc[len(df_error)] = [participant_id, "STEP01", e.cmd]

                    # Define STEP02: segment csv files for each participant id
                    if 2 in args.execute_steps:
                        _logger.info(f"STEP02: segment csv sensor files pipeline step for: {participant_id}")

                        try:
                            # Apply a synthetic signal desynchronize to some participants to study influence in metrics
                            if (desync_participant_ids is not None and participant_id in desync_participant_ids):
                                STEP02(args, df, participant_id, args.desync_segment_body, args.desync_seconds)
                            else:
                                STEP02(args, df, participant_id)                            
                        except subprocess.CalledProcessError as e:
                            df_error.loc[len(df_error)] = [participant_id, "STEP02", e.cmd]
                        
                    # Define STEP03: windowed segment files for convolution models for each participant id
                    if 3 in args.execute_steps:
                        _logger.info(f"STEP03: windowed segment files for convolution models pipeline step for: {participant_id}")

                        try:
                            STEP03(args,participant_id)
                        except subprocess.CalledProcessError as e:
                            df_error.loc[len(df_error)] = [participant_id, "STEP03", e.cmd]

                    # Define STEP04: extract features from windowed files for randomforest models for each participant id
                    if 4 in args.execute_steps:
                        _logger.info(f"STEP04: extract features from windowed files for randomforest models pipeline step for: {participant_id}")

                        try:
                            STEP04(args,participant_id)
                        except subprocess.CalledProcessError as e:
                            df_error.loc[len(df_error)] = [participant_id, "STEP04", e.cmd]

                    # Define STEP05: partial aggregation datasets for convolution and randomforest models for each participant id
                    if 5 in args.execute_steps:
                        _logger.info(f"STEP05: create participant windowed datasets pipeline step for: {participant_id}")

                        try:                        
                            STEP05(args, participant_id)
                        except subprocess.CalledProcessError as e:
                            df_error.loc[len(df_error)] = [participant_id, "STEP05", e.cmd]

        # Total datasets aggregation for all participants
        _logger.info("Total datasets aggregation for all participants")        
        if 6 in args.execute_steps:
            try: 
                STEP06(args)
            except subprocess.CalledProcessError as e:
                df_error.loc[len(df_error)] = [None, "STEP06", e.cmd]                

    df_error.to_csv("error_log.csv", index=False)

    _logger.info("Stop participant pre training")
    _logger.info("Pipeline ends here at " + str(datetime.now()))

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m wearablepermed_utils.skeleton 42
    #
    run()