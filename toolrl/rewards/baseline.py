# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# Modified: extract only the binary scorer, omit random debug printing.
import json
import re
from collections import Counter


def validate_result(result, answer):
    if len(result) == 0 or len(answer) == 0:
        return 2 if len(result) == len(answer) else 0
    try:
        counter1_full = Counter((item['name'], json.dumps(item['arguments'], sort_keys=True)) for item in result)
        counter2_full = Counter((item['name'], json.dumps(item['arguments'], sort_keys=True)) for item in answer)
    except TypeError:
        return 0
    if counter1_full == counter2_full:
        return 2
    if Counter(item['name'] for item in result) == Counter(item['name'] for item in answer):
        return 1
    return 0


def validate_format(tool_call_list):
    return all(isinstance(item, dict) and 'name' in item and 'arguments' in item for item in tool_call_list)


def extract_solution_v0(tool_call_str):
    marker = '<|im_start|>assistant'
    index = tool_call_str.rfind(marker)
    if index != -1:
        tool_call_str = tool_call_str[index:]
    matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, tool_call_str
    try:
        return json.loads(matches[-1].group(1).strip()), tool_call_str
    except json.JSONDecodeError:
        return None, tool_call_str


def compute_score_v0(solution_str, ground_truth):
    answer = json.loads(ground_truth)
    result, output_string = extract_solution_v0(solution_str)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = None
    if isinstance(result, dict):
        result = [result]
    if isinstance(answer, str):
        answer = json.loads(answer)
    if result is not None and ('<think>' not in output_string or '</think>' not in output_string):
        return 0
    if result is None or not validate_format(result):
        return 0
    return int(validate_result(result, answer) == 2)


def baseline_score(solution_str, ground_truth):
    try:
        return float(compute_score_v0(solution_str, ground_truth))
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError):
        return 0.0
