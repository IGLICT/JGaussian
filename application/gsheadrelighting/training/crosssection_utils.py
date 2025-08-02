# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import jittor as jt

def sample_cross_section(G, ws, resolution=256, w=1.2):
    axis=0
    # A, B = jt.meshgrid(jt.linspace(w/2, -w/2, resolution, device=ws.device), jt.linspace(-w/2, w/2, resolution, device=ws.device), indexing='ij')
    A, B = jt.meshgrid(jt.linspace(w/2, -w/2, resolution), jt.linspace(-w/2, w/2, resolution))
    A, B = A.reshape(-1, 1), B.reshape(-1, 1)
    C = jt.zeros_like(A)
    coordinates = [A, B]
    coordinates.insert(axis, C)
    coordinates = jt.cat(coordinates, dim=-1).expand(ws.shape[0], -1, -1)

    sigma = G.sample_mixed(coordinates, jt.randn_like(coordinates), ws)['sigma']
    return sigma.reshape(-1, 1, resolution, resolution)

# if __name__ == '__main__':
#     sample_crossection(None)