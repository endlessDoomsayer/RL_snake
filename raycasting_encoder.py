import numpy as np

class RayCastingEncoder:
    def __init__(self, num_rays=8):
        self.num_rays = num_rays
        self.angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        self.output_dim = num_rays * 3

    def encode(self, state_batch):
        # state_batch: (N, H, W, 4)
        N, H, W, _ = state_batch.shape
        results = np.zeros((N, self.num_rays, 3), dtype=np.float32)

        # Find all heads at once
        # head_mask is (N, H, W). argwhere gives [batch_idx, row, col]
        head_indices = np.argwhere(state_batch[..., 3] == 1.0)
        # Create a mapping for cases where a head might be missing (dead snake)
        heads = {idx[0]: (idx[1], idx[2]) for idx in head_indices}

        # Geometric Wall Distance
        # dr, dc shapes: (num_rays,)
        dr = -np.cos(self.angles)
        dc = np.sin(self.angles)

        for b_idx in range(N):
            if b_idx not in heads: continue
            hr, hc = heads[b_idx]
            
            # Distance to boundaries for all rays at once
            # row_dist: how many steps to hit top (0) or bottom (H-1)
            with np.errstate(divide='ignore', invalid='ignore'):
                t_row = np.where(dr < 0, hr / -dr, (H - 1 - hr) / dr)
                t_col = np.where(dc < 0, hc / -dc, (W - 1 - hc) / dc)
            
            # Wall distance is the first boundary hit + 1
            dist_wall = np.minimum(t_row, t_col) + 1
            results[b_idx, :, 0] = 1.0 / dist_wall

            # Fruit and Body
            board = state_batch[b_idx]
            # Get points for Fruit (channel 1) and Body (channel 2)
            for pt_type in [1, 2]:
                pts = np.argwhere(board[..., pt_type] == 1.0)
                if len(pts) == 0: continue
                
                # Vectors from head to all points
                v_r = pts[:, 0] - hr
                v_c = pts[:, 1] - hc
                dists = np.sqrt(v_r**2 + v_c**2)
                angles = np.arctan2(v_c, -v_r) % (2 * np.pi)
                
                # Assign to nearest ray
                angle_diffs = np.abs(angles[:, None] - self.angles[None, :])
                angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
                ray_bins = np.argmin(angle_diffs, axis=1)
                
                # Fill closest distance per ray
                for r_idx in range(self.num_rays):
                    mask = (ray_bins == r_idx)
                    if np.any(mask):
                        inv_d = 1.0 / np.min(dists[mask])
                        if inv_d > results[b_idx, r_idx, pt_type]:
                            results[b_idx, r_idx, pt_type] = inv_d

        return results.reshape(N, -1)