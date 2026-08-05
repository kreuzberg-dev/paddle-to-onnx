# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


import unittest

import hypothesis.strategies as st
import numpy as np
import paddle
from auto_scan_test import BaseNet, OPConvertAutoScanTest
from onnxbase import _test_with_pir, randtool


class Net(BaseNet):
    """
    Simple net for testing index_put
    """

    def forward(self, inputs, indices, value):
        accumulate = self.config.get("accumulate", False)
        indices = list(indices)
        return paddle.index_put(inputs, indices=indices, value=value, accumulate=accumulate)


class TestIndexPutConvert(OPConvertAutoScanTest):
    """
    api: paddle.index_put
    OPset version: 16
    """

    def sample_convert_config(self, draw):
        input_shape = draw(st.lists(st.integers(min_value=2, max_value=10), min_size=2, max_size=4))
        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        accumulate = draw(st.booleans())

        num_indices = draw(st.integers(min_value=1, max_value=len(input_shape)))

        def generator_indices():
            indices = []
            for i in range(num_indices):
                dim_limit = input_shape[i]
                indices.append(randtool("int", 0, dim_limit, shape=value_shape))
            return np.array(indices)

        value_shape = [
            input_shape[-1],
        ]

        def generator_value():
            return randtool("float", -5.0, 5.0, shape=value_shape).astype(dtype)

        config = {
            "op_names": ["index_put"],
            "test_data_shapes": [input_shape, generator_indices, generator_value],
            "test_data_types": [[dtype], ["int32"], [dtype]],
            "opset_version": [16],
            "input_spec_shape": [input_shape, [num_indices, *value_shape], value_shape],
            "accumulate": accumulate,
        }

        models = Net(config)

        return (config, models)

    @_test_with_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
