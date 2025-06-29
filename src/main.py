# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
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

from config import CFG, test_configurations
from tools.data_managers import print_and_train_log
from tools.training_initialization import run_single_training, run_test_configurations


def main():
    ##############################################################
    # TESTING mode
    ##############################################################
    if CFG.testing:
        print_and_train_log("Running in TESTING mode.\n", CFG.log_path)
        run_test_configurations(test_configurations)

    ##############################################################
    # SINGLE RUN mode
    ##############################################################
    else:
        # LOAD PREVIOUS MODEL:
        if CFG.load_timestamp:
            run_message = f"\nAttempting to load models from timestamp: {CFG.load_timestamp} and continue training...\n"
        # NEW RUN:
        else:
            run_message = "\nRunning with default configuration from config.py (new training)...\n"
        print_and_train_log(run_message, CFG.log_path)
        run_single_training()


if __name__ == "__main__":
    main()
