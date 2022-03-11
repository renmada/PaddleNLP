# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, default='./checkpoint/model_900/model_state.pdparams',
                    help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./output',
                    help="The path of model parameter in static graph to be saved.")
parser.add_argument("--memory_length", type=int, default=128,
                    help="Length of the retained previous heads.")
args = parser.parse_args()


# yapf: enable

def init_memory(batch_size, memory_length, d_model, n_layers):
    return [
        paddle.zeros(
            [batch_size, memory_length, d_model], dtype="float32")
        for _ in range(n_layers)
    ]


if __name__ == "__main__":
    # The number of labels should be in accordance with the training dataset.

    model = ppnlp.transformers.ErnieDocForSequenceClassification.from_pretrained(
        "ernie-doc-base-zh", num_classes=119)
    model_config = model.ernie_doc.config
    memory_input_spec = [
        paddle.static.InputSpec(shape=[None, args.memory_length, model_config['hidden_size']], dtype="float32")
        for _ in range(model_config['num_hidden_layers'])
    ]

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None, 1], dtype="int64"),  # input_ids
            *memory_input_spec,
            paddle.static.InputSpec(
                shape=[None, None, 1], dtype="int64"),  # token_type_ids
            paddle.static.InputSpec(
                shape=[None, None, 1], dtype="int64"),  # position_ids
            paddle.static.InputSpec(
                shape=[None, None, 1], dtype="float32"),  # attn_mask
        ])

    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
