# Copyright 2024 PennyLane Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from config import CFG
from runner.training_runner import run_single_training, run_test_configurations
from tools.data_managers import train_log
from training import Training  # Import the Training class


def main():
    parser = argparse.ArgumentParser(description="QGAN Subspace Training Script")
    parser.add_argument(
        "--load_timestamp",
        type=str,
        default=None,
        help="Timestamp of the model to load (e.g., 'YYYYMMDD_HHMMSS')",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Run with test configurations instead of the main configuration",
    )
    args = parser.parse_args()

    if args.testing:
        train_log("Running in TESTING mode.\n", CFG.get_log_path())
        run_test_configurations(CFG, train_log, Training, load_timestamp=args.load_timestamp)
    else:
        train_log("Running in NORMAL mode.\n", CFG.get_log_path())
        run_single_training(CFG, train_log, Training, load_timestamp=args.load_timestamp)


if __name__ == "__main__":
    main()
