"""
本地适配版：One-shot NPZ -> .pt cache for HDB 3D wind CFD data.
"""
import multiprocessing as mp
import os
import os.path as osp
import time
import json
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Worker globals (每个子进程初始化时赋予)
# ---------------------------------------------------------------------------
_W_DATA_DIR = None
_W_OUT_DIR = None
_W_SKIP_EXISTING = None

def _init_worker(data_dir, out_dir, skip_existing):
    global _W_DATA_DIR, _W_OUT_DIR, _W_SKIP_EXISTING
    _W_DATA_DIR = data_dir
    _W_OUT_DIR = out_dir
    _W_SKIP_EXISTING = skip_existing
    
    # 防止多进程内部再嵌套多线程导致抢夺 CPU
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def _worker_process(npz_basename):
    out_path = osp.join(_W_OUT_DIR, npz_basename.replace('.npz', '.pt'))
    if _W_SKIP_EXISTING and osp.exists(out_path):
        return ('skipped', npz_basename, None)
    try:
        case = process_one_npz(osp.join(_W_DATA_DIR, npz_basename))
        torch.save(case, out_path)
        return ('ok', npz_basename, None)
    except Exception as exc:                                      
        return ('failed', npz_basename, str(exc))

# ---------------------------------------------------------------------------
# 核心数据处理逻辑 (未修改，保留原汁原味)
# ---------------------------------------------------------------------------
def process_one_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    # STL geometry 
    stl_vertices = torch.from_numpy(np.ascontiguousarray(data['stl_coordinates'])).float() 
    stl_centers  = torch.from_numpy(np.ascontiguousarray(data['stl_centers'])).float()       
    stl_faces    = torch.from_numpy(np.ascontiguousarray(data['stl_faces'])).long()          
    stl_areas    = torch.from_numpy(np.ascontiguousarray(data['stl_areas'])).float()         

    # Surface points 
    surface_pos     = torch.from_numpy(np.ascontiguousarray(data['surface_mesh_centers'])).float() 
    surface_normals = torch.from_numpy(np.ascontiguousarray(data['surface_normals'])).float()      
    surface_areas   = torch.from_numpy(np.ascontiguousarray(data['surface_areas'])).float()        
    surface_fields  = torch.from_numpy(np.ascontiguousarray(data['surface_fields'])).float()       

    # Volume CFD field 
    volume_pos    = torch.from_numpy(np.ascontiguousarray(data['volume_mesh_centers'])).float() 
    volume_fields = torch.from_numpy(np.ascontiguousarray(data['volume_fields'])).float()       

    # Global parameters 
    global_params = torch.from_numpy(np.ascontiguousarray(data['global_params_values'])).float() 

    # --- 最耗时的步骤：计算 SDF ---
    tree = cKDTree(stl_centers.numpy())
    sdf_dist, _ = tree.query(volume_pos.numpy(), k=1, workers=1)
    volume_sdf = torch.from_numpy(sdf_dist.astype(np.float32))   

    stl_centroid = stl_centers.mean(dim=0)                       

    volume_bbox_min = volume_pos.min(dim=0).values               
    volume_bbox_max = volume_pos.max(dim=0).values               
    stl_bbox_min    = stl_centers.min(dim=0).values              
    stl_bbox_max    = stl_centers.max(dim=0).values              

    return {
        'case_name': osp.basename(npz_path).replace('.npz', ''),
        'stl_vertices': stl_vertices,
        'stl_centers': stl_centers,
        'stl_faces': stl_faces,
        'stl_areas': stl_areas,
        'stl_centroid': stl_centroid,
        'stl_bbox_min': stl_bbox_min,
        'stl_bbox_max': stl_bbox_max,
        'surface_pos': surface_pos,
        'surface_normals': surface_normals,
        'surface_areas': surface_areas,
        'surface_fields': surface_fields,   
        'volume_pos': volume_pos,
        'volume_fields': volume_fields,     
        'volume_sdf': volume_sdf,           
        'volume_bbox_min': volume_bbox_min,
        'volume_bbox_max': volume_bbox_max,
        'global_params': global_params,     
    }

# ---------------------------------------------------------------------------
# Driver (修改为直接读取本地固定路径)
# ---------------------------------------------------------------------------
def main():
    # ================= 修改这里：你本地的参数配置 =================
    # 你的文件存放路径 (输入和输出放在同一个文件夹)
    WORK_DIR = r"C:\Users\3017y\Downloads\streamline"
    NUM_WORKERS = 10
    SKIP_EXISTING = True
    # ==============================================================

    os.makedirs(WORK_DIR, exist_ok=True)

    # 1. 找到所有 npz 文件 (包括你之前生成的 _filtered.npz)
    all_npz = sorted([f for f in os.listdir(WORK_DIR) if f.endswith('_filtered.npz')])
    print(f"[preprocess_hdb] Found {len(all_npz)} NPZ files in {WORK_DIR}")
    
    if not all_npz:
        print("没有找到 .npz 文件，请检查路径！")
        return

    to_process = all_npz

    # 2. 开启多进程处理
    t_start = time.time()
    statuses = {'ok': 0, 'skipped': 0, 'failed': 0}
    failures = []

    print(f"🚀 启动！使用 {NUM_WORKERS} 个核心并行处理...")

    # 在 Windows 上直接用 mp.Pool 即可，它会自动使用安全的 spawn 模式
    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(WORK_DIR, WORK_DIR, SKIP_EXISTING),
    ) as pool:
        for status, cname, msg in tqdm(
            pool.imap_unordered(_worker_process, to_process, chunksize=1),
            total=len(to_process),
            desc='正在转换 .pt 文件',
        ):
            statuses[status] += 1
            if status == 'failed':
                print(f"\n!! failed {cname}: {msg}")
                failures.append((cname, msg))

    elapsed = time.time() - t_start

    # 3. 打印总结
    print(f"\n✅ [preprocess_hdb] DONE in {elapsed:.0f}s "
          f"(ok={statuses['ok']}, skipped={statuses['skipped']}, failed={statuses['failed']})")
    
    cache_count = len([f for f in os.listdir(WORK_DIR) if f.endswith('.pt')])
    print(f"[preprocess_hdb] 目前 {WORK_DIR} 文件夹里共有 {cache_count} 个 .pt 缓存文件。")

if __name__ == '__main__':
    main()