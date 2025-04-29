# rng_init.py
import numpy as np
import random, torch, os

def rng_worker_init(base_seed: int):
    """
    Pool 이 fork/spawn 한 '각 프로세스'에서 단 한 번 호출된다.
    base_seed : run 전체에 동일한 값
    """
    pid   = os.getpid()          # 고유 프로세스 ID
    seed  = base_seed + pid      # 서로 다른 고유 시드

    # Python 표준 RNG
    random.seed(seed)

    # NumPy: Philox 는 병렬 친화
    global np_rng                # ===> worker 전역으로 노출
    np_rng = np.random.default_rng(seed)

    # Torch
    seed = seed  % (2 ** 64)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # GPU 사용 시
    global torch_rng
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)
