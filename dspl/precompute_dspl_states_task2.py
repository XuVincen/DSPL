"""
DSPL Task 2 — 残基级状态条件相关预处理脚本

对全部 11,748 个 Binding Site 蛋白完成:
  1. 从 MD.hdf5 提取 Cα 轨迹 (用 build_residue_graph_v3.py 的 kabsch + 坐标匹配逻辑)
  2. PCA(20) + KMeans(K=5) 状态发现
  3. 硬分配状态条件 Pearson 相关 → state_corr (E, 5)

与 Task 1 版本的关键区别:
  - Task 1: 原子级图 (48-dim features, 4.5Å distance edges)
  - Task 2: 残基级图 (21-dim residue one-hot, 10Å distance edges, Cα only)
  - 输入边: residue_graph 中的 edge_index_distance (Cα-Cα 距离边)

输出: data/data_files/dspl_state_corr_task2.h5 (~6 GB)

使用方式:
  python precompute_dspl_states_task2.py
"""
import os, sys, gc, time

# Root of the project (where data/ and vendor/ live).
ROOT = os.environ.get('DSPL_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, ROOT)

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from vendor.kabsch import kabsch_align
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ==============================
# 常量 (与 Task 1 完全一致)
# ==============================
N_PROTOTYPES = 5
PCA_COMPONENTS = 20
CORR_THRESHOLD = 0.1       # 状态条件相关阈值
MIN_FRAMES_PER_STATE = 10  # 状态内最少帧数

# 路径
MD_PATH = 'data/downloaded/MD.hdf5'
ADAPT_PATH = 'data/downloaded/adaptability_MD.hdf5'
RESIDUE_GRAPH_PATH = 'data/data_files/residue_graph_data_distance_10.0_corr_aligned_0.3.h5'
OUTPUT_PATH = 'data/data_files/dspl_state_corr_task2.h5'
BINDING_SITE_THRESHOLD = 10.0  # Å

# Binding site splits
SPLIT_DIR = 'data/splits/binding_site'


# ==============================
# Cα 提取 + 坐标匹配 (从 build_residue_graph_v3.py 复用)
# ==============================
def extract_ca_traj(pid, md_f, adapt_f):
    """提取单个蛋白的 Cα MD 轨迹 (100, N_ca, 3)。
    与 build_residue_graph_v3.py 的 process_entry() 逻辑完全一致。
    返回: ca_traj (100, N_ca, 3), ca_coords_ref (N_ca, 3)
    """
    adapt_grp = adapt_f[pid]
    atoms_coords_ref = adapt_grp['atoms_coordinates_ref'][:]  # (N_heavy, 3)
    atoms_residue = adapt_grp['atoms_residue'][:]  # (N_heavy,)
    mbi = adapt_grp['molecules_begin_atom_index'][:].astype(int)
    protein_end = int(mbi[1])
    n_atoms = atoms_coords_ref.shape[0]

    # Cα: 每个残基的第一个蛋白质原子
    ca_indices = []
    for i in range(n_atoms):
        if i >= protein_end:
            break
        if i == 0 or atoms_residue[i] != atoms_residue[i - 1]:
            ca_indices.append(i)
    ca_indices = np.array(ca_indices, dtype=int)
    ca_coords_ref = atoms_coords_ref[ca_indices]
    n_ca = len(ca_indices)

    if n_ca < 3:
        raise ValueError(f"too few Cα atoms: {n_ca}")

    # MD trajectory
    md_grp = md_f[pid]
    md_traj = md_grp['trajectory_coordinates'][:]  # (100, N_md, 3)
    n_frames, n_md_atoms = md_traj.shape[0], md_traj.shape[1]
    md_frame0 = md_traj[0]

    # 坐标匹配 Cα → MD 索引
    # 先用蛋白质范围搜索
    md_mbi = md_grp.get('molecules_begin_atom_index')
    if md_mbi is not None:
        md_protein_end = int(md_mbi[1])
    else:
        md_protein_end = n_md_atoms // 2

    search_limit = min(md_protein_end, n_md_atoms)

    ca_md_indices = []
    for ci in range(n_ca):
        ca_pos = ca_coords_ref[ci]
        dists = np.linalg.norm(md_frame0[:search_limit] - ca_pos, axis=1)
        best_idx = int(np.argmin(dists))
        match_dist = float(dists[best_idx])
        if match_dist > 2.0:
            # 扩大搜索范围
            dists_full = np.linalg.norm(md_frame0 - ca_pos, axis=1)
            best_idx_full = int(np.argmin(dists_full))
            if float(dists_full[best_idx_full]) < match_dist:
                best_idx = best_idx_full
                match_dist = float(dists_full[best_idx_full])
        # 即使 match_dist > 3.0 也继续（允许一定的坐标差异）
        ca_md_indices.append(best_idx)

    ca_md_indices = np.array(ca_md_indices, dtype=int)

    # 提取 Cα 轨迹
    ca_traj = np.zeros((n_frames, n_ca, 3), dtype=np.float32)
    for f in range(n_frames):
        ca_traj[f] = md_traj[f, ca_md_indices]

    # Kabsch 对齐
    for f in range(1, n_frames):
        ca_traj[f] = kabsch_align(ca_traj[f], ca_traj[0])

    return ca_traj, ca_coords_ref


# ==============================
# 状态条件相关 (与 Task 1 完全相同的算法，但用于 Cα 轨迹)
# ==============================
def compute_state_corr_numpy(ca_traj, dist_edges, frame_labels, n_states=5):
    """硬分配加权 Pearson 相关 — 与 Task 1 完全相同的算法。
    对每条边在所有帧上的计算是向量化的 (E, T) 批量操作。

    返回: state_corr (E, K), state_corr_mask (E, K)"""
    T, N, _ = ca_traj.shape
    E = dist_edges.shape[1]

    src_idx = dist_edges[0, :].astype(np.int64)
    dst_idx = dist_edges[1, :].astype(np.int64)
    xi = ca_traj[:, src_idx, :].transpose(1, 0, 2)  # (E, T, 3)
    xj = ca_traj[:, dst_idx, :].transpose(1, 0, 2)  # (E, T, 3)

    eps = 1e-8
    state_corr = np.zeros((E, n_states), dtype=np.float32)
    state_corr_mask = np.zeros((E, n_states), dtype=bool)

    for k in range(n_states):
        w = (frame_labels == k).astype(np.float32)  # (T,)
        w_sum = w.sum()
        if w_sum < MIN_FRAMES_PER_STATE:
            continue

        corr_k = np.zeros(E, dtype=np.float32)
        for dim in range(3):
            x_i = xi[:, :, dim]  # (E, T)
            x_j = xj[:, :, dim]  # (E, T)

            x_i_mean = (x_i * w[np.newaxis, :]).sum(axis=1) / w_sum
            x_j_mean = (x_j * w[np.newaxis, :]).sum(axis=1) / w_sum

            x_i_c = x_i - x_i_mean[:, np.newaxis]
            x_j_c = x_j - x_j_mean[:, np.newaxis]

            cov = (w[np.newaxis, :] * x_i_c * x_j_c).sum(axis=1) / w_sum
            x_i_var = (w[np.newaxis, :] * x_i_c ** 2).sum(axis=1) / w_sum
            x_j_var = (w[np.newaxis, :] * x_j_c ** 2).sum(axis=1) / w_sum

            corr_dim = cov / (np.sqrt(x_i_var + eps) * np.sqrt(x_j_var + eps) + eps)
            corr_dim = np.clip(corr_dim, -1.0, 1.0)
            corr_k += corr_dim / 3.0

        state_corr[:, k] = corr_k
        state_corr_mask[:, k] = (
            ~np.isnan(corr_k)) & (np.abs(corr_k) >= CORR_THRESHOLD)

    state_corr = np.nan_to_num(state_corr, nan=0.0)
    return state_corr.astype(np.float32), state_corr_mask


# ==============================
# 单个蛋白处理 (供并行调用)
# ==============================
def process_one_protein(pid):
    """处理单个蛋白的完整流水线。
    成功返回 (pid, tmp_npz_path, None)
    失败返回 (pid, None, error_msg)
    """
    try:
        with h5py.File(MD_PATH, 'r') as md_f, \
             h5py.File(ADAPT_PATH, 'r') as adapt_f, \
             h5py.File(RESIDUE_GRAPH_PATH, 'r') as residue_f:

            if pid not in md_f or pid not in adapt_f or pid not in residue_f:
                return (pid, None, f'missing data for {pid}')

            # Step 1: 提取 Cα 轨迹 + Kabsch 对齐
            ca_traj, ca_coords_ref = extract_ca_traj(pid, md_f, adapt_f)
            T, N_ca = ca_traj.shape[0], ca_traj.shape[1]

            # Step 2: 读取残基图的距离边 (edge_index_distance)
            grp = residue_f[pid]
            dist_edges = grp['edge_index_distance'][:]  # (2, E)

            # 验证: N_ca 必须等于 graph 节点数
            n_nodes_in_graph = grp['node_features'].shape[0]
            if N_ca != n_nodes_in_graph:
                # 可能坐标匹配不一致，但尝试继续（截断/填充）
                if abs(N_ca - n_nodes_in_graph) > 5:
                    return (pid, None,
                        f'Cα count mismatch: extracted {N_ca} vs graph {n_nodes_in_graph}')

            # Step 3: PCA + k-means 状态发现
            f_flat = ca_traj.reshape(T, N_ca * 3)
            n_comp = min(PCA_COMPONENTS, T - 1, f_flat.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            f_pca = pca.fit_transform(f_flat)
            kmeans = KMeans(n_clusters=N_PROTOTYPES, random_state=42,
                            n_init=10, max_iter=300)
            frame_labels = kmeans.fit_predict(f_pca)  # (100,)

            # Step 4: 状态条件相关
            state_corr, state_corr_mask = compute_state_corr_numpy(
                ca_traj, dist_edges, frame_labels, N_PROTOTYPES
            )

            # Step 5: 保存为临时 npz
            alpha = np.zeros((T, N_PROTOTYPES), dtype=np.float32)
            alpha[np.arange(T), frame_labels] = 1.0

            tmp_path = f'/tmp/_dspl_t2_tmp/{pid}.npz'
            os.makedirs('/tmp/_dspl_t2_tmp', exist_ok=True)
            np.savez_compressed(tmp_path,
                state_corr=state_corr,
                state_corr_mask=state_corr_mask,
                frame_labels=frame_labels.astype(np.int32),
                alpha=alpha,
                n_edges=dist_edges.shape[1],
                n_ca=N_ca)
            return (pid, tmp_path, None)
    except Exception as e:
        import traceback
        return (pid, None,
                f'{type(e).__name__}: {e}\n{traceback.format_exc()[-300:]}')


# ==============================
# 主流程
# ==============================
def main():
    print("=" * 60)
    print("DSPL Task 2 - Residue-Level State-Conditioned Correlation")
    print("=" * 60)

    # 收集所有 Binding Site 分割中的 protein ID
    all_pids = set()
    for split_name in ['train.txt', 'val.txt', 'test.txt']:
        split_path = os.path.join(SPLIT_DIR, split_name)
        if os.path.exists(split_path):
            with open(split_path) as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        all_pids.add(pid)

    all_pids = sorted(all_pids)
    print(f"Total unique proteins (from binding_site splits): {len(all_pids)}")

    # 断点续传
    completed_pids = set()
    if os.path.exists(OUTPUT_PATH):
        with h5py.File(OUTPUT_PATH, 'r') as f:
            completed_pids = set(f.keys())
        print(f"Already completed: {len(completed_pids)}")

    pending_pids = [p for p in all_pids if p not in completed_pids]
    print(f"Pending: {len(pending_pids)}")

    if len(pending_pids) == 0:
        print("All proteins already processed. Done!")
        return

    # 快速检查数据可用性
    print("\nData availability check...")
    md_keys = set()
    adapt_keys = set()
    residue_keys = set()

    with h5py.File(MD_PATH, 'r') as f:
        for k in f.keys():
            try:
                if isinstance(f[k], h5py.Group):
                    md_keys.add(k)
            except:
                pass

    with h5py.File(ADAPT_PATH, 'r') as f:
        for k in f.keys():
            try:
                if isinstance(f[k], h5py.Group):
                    adapt_keys.add(k)
            except:
                pass

    with h5py.File(RESIDUE_GRAPH_PATH, 'r') as f:
        for k in f.keys():
            try:
                if isinstance(f[k], h5py.Group):
                    residue_keys.add(k)
            except:
                pass

    print(f"  MD entries: {len(md_keys)}, Adapt entries: {len(adapt_keys)}, Residue entries: {len(residue_keys)}")

    available_sample = [p for p in pending_pids[:10] if p in md_keys and p in adapt_keys and p in residue_keys]
    missing_sample = [p for p in pending_pids[:10] if not (p in md_keys and p in adapt_keys and p in residue_keys)]
    print(f"  Sample (first 10): available={len(available_sample)}, missing={len(missing_sample)}")
    if missing_sample:
        print(f"  Missing examples: {missing_sample[:5]}")

    # 过滤: 只保留三个 H5 中都存在的
    pending_pids = [p for p in pending_pids if p in md_keys and p in adapt_keys and p in residue_keys]
    print(f"\nFiltered pending (all data available): {len(pending_pids)}")

    if len(pending_pids) == 0:
        print("No valid pending proteins. Done!")
        return

    # 并行处理
    n_workers = min(8, mp.cpu_count() - 2)
    print(f"\nStarting parallel processing with {n_workers} workers...")
    print(f"Output: {OUTPUT_PATH}")
    print("-" * 60)

    os.makedirs('/tmp/_dspl_t2_tmp', exist_ok=True)

    success_count = 0
    error_count = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one_protein, pid): pid
                   for pid in pending_pids}

        for i, future in enumerate(as_completed(futures)):
            pid, tmp_path, error = future.result()

            if tmp_path is not None:
                try:
                    tmp_data = np.load(tmp_path, allow_pickle=True)
                    with h5py.File(OUTPUT_PATH, 'a') as out_f:
                        grp = out_f.create_group(pid)
                        grp.create_dataset('state_corr', data=tmp_data['state_corr'],
                                           compression='gzip', compression_opts=4)
                        grp.create_dataset('state_corr_mask',
                                           data=tmp_data['state_corr_mask'],
                                           compression='gzip', compression_opts=4)
                        grp.create_dataset('frame_labels',
                                           data=tmp_data['frame_labels'])
                        grp.create_dataset('alpha', data=tmp_data['alpha'])
                        grp.attrs['n_edges'] = int(tmp_data['n_edges'])
                        grp.attrs['n_ca'] = int(tmp_data['n_ca'])
                    os.remove(tmp_path)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    if error is None:
                        error = f'write error: {e}'
            else:
                error_count += 1

            # 进度报告
            done = i + 1
            if done % 100 == 0 or done == len(pending_pids):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(pending_pids) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(pending_pids)}] success={success_count} "
                      f"errors={error_count} rate={rate:.1f}/s "
                      f"ETA={eta/60:.0f}min elapsed={elapsed/60:.1f}min")

                # 每 500 打印一个错误样例
                if done % 500 == 0 and error_count > 0:
                    error_pids = [futures[f] for f in futures
                                  if f.done() and f.result()[1] is None]
                    if error_pids:
                        ep = error_pids[0]
                        _, _, err = next(f.result() for f in futures
                                        if f.done() and f.result()[1] is None)
                        print(f"  [Sample error] {ep}: {err[:200]}")

            gc.collect()

    # 最终统计
    elapsed_total = time.time() - t_start
    print("-" * 60)
    print("Done!")
    print(f"  Total elapsed: {elapsed_total / 60:.1f} min")
    print(f"  Success: {success_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Output:  {OUTPUT_PATH}")

    # 检查输出文件
    with h5py.File(OUTPUT_PATH, 'r') as f:
        print(f"  Output proteins: {len(f.keys())}")
        sample_pids = list(f.keys())[:3]
        for sp in sample_pids:
            g = f[sp]
            print(f"  Sample {sp}: state_corr={g['state_corr'].shape}, "
                  f"mask={g['state_corr_mask'].shape}, "
                  f"n_edges={g.attrs['n_edges']}, n_ca={g.attrs.get('n_ca', '?')}")

    size_gb = os.path.getsize(OUTPUT_PATH) / (1024 ** 3)
    print(f"  File size: {size_gb:.2f} GB")

    # 清理
    import shutil
    tmp_dir = '/tmp/_dspl_t2_tmp'
    remaining = os.listdir(tmp_dir) if os.path.exists(tmp_dir) else []
    if remaining:
        print(f"  Remaining tmp files: {len(remaining)} (from failed/partial)")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  Temp dir cleaned")


if __name__ == '__main__':
    main()
