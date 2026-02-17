# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch.nn as nn

class GASDSACV2Learner(nn.Module):
    """
    [LEGACY] This class is no longer used. 
    Please use train_gasd_sac.py which directly employs the new SACV2Learner.
    """
    def __init__(self, **kwargs):
        super(GASDSACV2Learner, self).__init__()
        print("[Warning] Attempted to use legacy GASDSACV2Learner. This is no longer supported.")
