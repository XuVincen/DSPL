"""
DSPL Task 1 — 离线状态条件相关预处理脚本

对全部 13,790 蛋白完成:
  1. 坐标匹配 (cKDTree) — 防 index 不匹配
  2. Kabsch 对齐
  3. PCA(20) + KMeans(K=5) 状态发现
  4. 硬分配状态条件 Pearson 相关 → state_corr (E, 5)

输出: data/data_files/dspl_state_corr_task1.h5 (~10 GB)

并行: ProcessPoolExecutor(max_workers=8)
断点续传: 检查输出 H5 中已存在的 pid，跳过已完成项

使用方式:
  python precompute_dspl_states.py
"""
import os, sys, gc, time, traceback

# Root of the project (where data/ and vendor/ live).
ROOT = os.environ.get('DSPL_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, ROOT)

import h5py
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from vendor.kabsch import kabsch_align
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ==============================
# 常量 (与 train_ablation.py 完全一致)
# ==============================
N_PROTOTYPES = 5
PCA_COMPONENTS = 20
CORR_THRESHOLD = 0.1       # 状态条件相关阈值 (与 train_ablation.py 一致)
MIN_FRAMES_PER_STATE = 10  # 状态内最少帧数

# 路径
MD_PATH = 'data/downloaded/MD.hdf5'
ATOM_GRAPH_PATH = 'data/data_files/atom_graph_OnlyProtein_distance_4.5_planA.h5'
OUTPUT_PATH = 'data/data_files/dspl_state_corr_task1.h5'
SPLIT_DIR = 'data/splits/adaptability'


# ==============================
# 坐标匹配 + Kabsch 对齐 (复用 train_ablation.py preload_md 逻辑)
# ==============================
def load_and_align(pid, md_file, atom_h5_f):
    """对单个蛋白: KDTree匹配 + 提取轨迹 + Kabsch对齐。
    与 train_ablation.py preload_md() 第240-258行完全一致。
    返回: atom_traj numpy (100, N_graph, 3) 或 None"""
    grp = md_file[pid]
    mbi = grp['molecules_begin_atom_index'][:]
    p_start, p_end = int(mbi[0]), int(mbi[1])
    full_traj = grp['trajectory_coordinates'][:]  # (100, N_md, 3)
    f0 = full_traj[0, p_start:p_end]              # 仅蛋白质原子

    graph_coords = atom_h5_f[pid]['atoms_coordinates'][:]  # (N_graph, 3)
    tree = cKDTree(f0)
    _, nn_idx = tree.query(graph_coords, k=1)
    atom_idx = nn_idx + p_start

    atom_traj = full_traj[:, atom_idx].copy()  # (100, N_graph, 3)
    ref = atom_traj[0].copy()
    for f in range(1, atom_traj.shape[0]):
        atom_traj[f] = kabsch_align(atom_traj[f], ref)

    # 验证: atom_traj.shape[1] 必须等于 graph 节点数
    if atom_traj.shape[1] != graph_coords.shape[0]:
        raise ValueError(
            "atom count mismatch: {} vs {}".format(
                atom_traj.shape[1], graph_coords.shape[0]))
    return atom_traj.astype(np.float32)


# ==============================
# 状态条件相关 (硬分配, 向量化, 数值等价于 _compute_state_corr)
# ==============================
def compute_state_corr_numpy(atom_traj, dist_edges, frame_labels, n_states=5):
    """硬分配加权 Pearson 相关 — numpy 后端, 向量化实现。
    与 train_ablation.py _compute_state_corr() 数值等价。
    对每条边在所有帧上的计算是向量化的 (E, T) 批量操作,
    仅在 K=5 个状态和 3 个维度上循环 (15次迭代)。

    返回: state_corr (E, K), state_corr_mask (E, K)"""
    T, N, _ = atom_traj.shape
    E = dist_edges.shape[1]

    # 提取所有边涉及的原子对 (向量化): (E, T) x 3 维
    src_idx = dist_edges[0, :].astype(np.int64)  # (E,)
    dst_idx = dist_edges[1, :].astype(np.int64)  # (E,)
    # atom_traj: (T, N, 3) → 按边索引: (E, T, 3)
    xi = atom_traj[:, src_idx, :].transpose(1, 0, 2)  # (E, T, 3)
    xj = atom_traj[:, dst_idx, :].transpose(1, 0, 2)  # (E, T, 3)

    eps = 1e-8
    state_corr = np.zeros((E, n_states), dtype=np.float32)
    state_corr_mask = np.zeros((E, n_states), dtype=bool)

    for k in range(n_states):
        w = (frame_labels == k).astype(np.float32)  # (T,)
        w_sum = w.sum()
        if w_sum < MIN_FRAMES_PER_STATE:
            continue  # 全零 (mask 保持 False)

        corr_k = np.zeros(E, dtype=np.float32)
        for dim in range(3):
            x_i = xi[:, :, dim]  # (E, T)
            x_j = xj[:, :, dim]  # (E, T)

            # 加权均值: (E,)
            x_i_mean = (x_i * w[np.newaxis, :]).sum(axis=1) / w_sum
            x_j_mean = (x_j * w[np.newaxis, :]).sum(axis=1) / w_sum

            # 中心化: (E, T)
            x_i_c = x_i - x_i_mean[:, np.newaxis]
            x_j_c = x_j - x_j_mean[:, np.newaxis]

            # 加权协方差: (E,)
            cov = (w[np.newaxis, :] * x_i_c * x_j_c).sum(axis=1) / w_sum

            # 加权方差: (E,)
            x_i_var = (w[np.newaxis, :] * x_i_c ** 2).sum(axis=1) / w_sum
            x_j_var = (w[np.newaxis, :] * x_j_c ** 2).sum(axis=1) / w_sum

            # Pearson: (E,)
            corr_dim = cov / (np.sqrt(x_i_var + eps) * np.sqrt(x_j_var + eps) + eps)
            corr_dim = np.clip(corr_dim, -1.0, 1.0)
            corr_k += corr_dim / 3.0

        state_corr[:, k] = corr_k
        state_corr_mask[:, k] = (
            ~np.isnan(corr_k)) & (np.abs(corr_k) >= CORR_THRESHOLD)

    # Fill NaN with 0
    state_corr = np.nan_to_num(state_corr, nan=0.0)

    return state_corr.astype(np.float32), state_corr_mask


# ==============================
# 单个蛋白处理 (供并行调用)
# ==============================
def process_one_protein(pid):
    """处理单个蛋白的完整流水线。
    成功返回 (pid, tmp_npz_path, None)
    失败返回 (pid, None, error_msg)
    每个 worker 独立打开 H5 (只读)，结果通过临时 npz 文件传回主进程。"""
    try:
        with h5py.File(MD_PATH, 'r') as md_file, \
             h5py.File(ATOM_GRAPH_PATH, 'r') as atom_h5_f:

            if pid not in md_file or pid not in atom_h5_f:
                return (pid, None, 'missing in MD or atom_graph H5')

            # Step 1: 坐标匹配 + Kabsch
            atom_traj = load_and_align(pid, md_file, atom_h5_f)

            # Step 2: 读取距离边
            dist_edges = atom_h5_f[pid]['edge_index_distance'][:]  # (2, E)

            # Step 3: PCA + k-means 状态发现
            T, N, _ = atom_traj.shape
            f_flat = atom_traj.reshape(T, N * 3)
            n_comp = min(PCA_COMPONENTS, T - 1, f_flat.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            f_pca = pca.fit_transform(f_flat)
            kmeans = KMeans(n_clusters=N_PROTOTYPES, random_state=42,
                            n_init=10, max_iter=300)
            frame_labels = kmeans.fit_predict(f_pca)  # (100,)

            # Step 4: 状态条件相关
            state_corr, state_corr_mask = compute_state_corr_numpy(
                atom_traj, dist_edges, frame_labels, N_PROTOTYPES
            )

            # Step 5: 保存为临时 npz (避免跨进程传输大数据)
            alpha = np.zeros((T, N_PROTOTYPES), dtype=np.float32)
            alpha[np.arange(T), frame_labels] = 1.0

            tmp_path = '/tmp/_dspl_tmp/{}.npz'.format(pid)
            os.makedirs('/tmp/_dspl_tmp', exist_ok=True)
            np.savez_compressed(tmp_path,
                state_corr=state_corr,
                state_corr_mask=state_corr_mask,
                frame_labels=frame_labels.astype(np.int32),
                alpha=alpha,
                n_edges=dist_edges.shape[1])
            return (pid, tmp_path, None)
    except Exception as e:
        return (pid, None,
                '{}: {}'.format(type(e).__name__, str(e)[:200]))


# ==============================
# 主流程
# ==============================
def main():
    print("=" * 60)
    print("DSPL Task 1 - State-Conditioned Correlation Precomputation")
    print("=" * 60)

    # 收集所有 protein ID
    all_pids = set()
    for split_name in ['train.txt', 'val.txt', 'test.txt']:
        split_path = os.path.join(SPLIT_DIR, split_name)
        with open(split_path) as f:
            for line in f:
                pid = line.strip()
                if pid:
                    all_pids.add(pid)

    all_pids = sorted(all_pids)
    print("Total unique proteins: {}".format(len(all_pids)))

    # 断点续传: 检查已完成的 pid
    completed_pids = set()
    if os.path.exists(OUTPUT_PATH):
        with h5py.File(OUTPUT_PATH, 'r') as f:
            completed_pids = set(f.keys())
        print("Already completed: {}".format(len(completed_pids)))

    pending_pids = [p for p in all_pids if p not in completed_pids]
    print("Pending: {}".format(len(pending_pids)))

    if len(pending_pids) == 0:
        print("All proteins already processed. Done!")
        return

    # 检查数据可用性
    print("\nChecking data availability...")
    missing_md = 0
    missing_graph = 0
    with h5py.File(MD_PATH, 'r') as md_f, \
         h5py.File(ATOM_GRAPH_PATH, 'r') as ag_f:
        for pid in pending_pids[:100]:  # 抽样检查前100
            if pid not in md_f:
                missing_md += 1
            if pid not in ag_f:
                missing_graph += 1
    print("  Sample check (first 100): missing MD={}, missing graph={}".format(
        missing_md, missing_graph))

    # 并行处理
    n_workers = min(8, mp.cpu_count() - 2)
    print("\nStarting parallel processing with {} workers...".format(n_workers))
    print("Output: {}".format(OUTPUT_PATH))
    print("-" * 60)

    # 创建临时目录
    os.makedirs('/tmp/_dspl_tmp', exist_ok=True)

    success_count = 0
    error_count = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one_protein, pid): pid
                   for pid in pending_pids}

        for i, future in enumerate(as_completed(futures)):
            pid, tmp_path, error = future.result()

            if tmp_path is not None:
                # 从临时文件读取并写入输出 H5（主进程单线程写入，安全）
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
                # 清理临时文件
                os.remove(tmp_path)
                success_count += 1
            else:
                error_count += 1

            # 进度报告
            done = i + 1
            if done % 100 == 0 or done == len(pending_pids):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(pending_pids) - done) / rate if rate > 0 else 0
                print("  [{}/{}] success={} errors={} rate={:.1f}/s ETA={:.0f}min elapsed={:.1f}min".format(
                    done, len(pending_pids), success_count, error_count,
                    rate, eta / 60, elapsed / 60))

            gc.collect()

    # 最终统计
    elapsed_total = time.time() - t_start
    print("-" * 60)
    print("Done!")
    print("  Total elapsed: {:.1f} min".format(elapsed_total / 60))
    print("  Success: {}".format(success_count))
    print("  Errors:  {}".format(error_count))
    print("  Output:  {}".format(OUTPUT_PATH))

    # 检查输出文件
    with h5py.File(OUTPUT_PATH, 'r') as f:
        print("  Output proteins: {}".format(len(f.keys())))
        # 抽样检查
        sample_pids = list(f.keys())[:3]
        for sp in sample_pids:
            g = f[sp]
            print("  Sample {}: state_corr={}, mask={}, n_edges={}".format(
                sp, g['state_corr'].shape, g['state_corr_mask'].shape,
                g.attrs['n_edges']))

    # 文件大小
    size_gb = os.path.getsize(OUTPUT_PATH) / (1024 ** 3)
    print("  File size: {:.2f} GB".format(size_gb))

    # 清理临时目录
    import shutil
    tmp_dir = '/tmp/_dspl_tmp'
    remaining = os.listdir(tmp_dir) if os.path.exists(tmp_dir) else []
    if remaining:
        print("  Remaining tmp files: {} (from failed/partial)".format(
            len(remaining)))
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  Temp dir cleaned")


if __name__ == '__main__':
    main()
