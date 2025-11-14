#!/bin/bash


#SBATCH -p apu
#SBATCH -t 0:15:00
#SBATCH -o bind-%j.out
#SBATCH -e bind-%j.error
#SBATCH --gres gpu:2


# Lightning DDP configuration following official docs
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --reservation=amd




# ================================================================
# Experiments:
#   A) hello_jobstep-style (srun; HIP + OpenMP): OMP_NUM_THREADS=1,2,4
#      - Dynamically prints RT_GPU_ID (HIP logical id), GPU_ID (DRM cardN), Bus_ID
#      - Per-thread NUMA node via optional libnuma (dlopen at runtime; prints -1 if not available)
#   B) pthreads affinity (no OpenMP): THREADS=1,2,4
#      - Prints per-thread CPU, NUMA node (optional libnuma), and CPU affinity mask
#      - Uses srun only if SLURM_NTASKS > 1
#   C) OpenMP affinity: OMP_NUM_THREADS=1,2,4
#      - Prints per-thread CPU, NUMA node (optional libnuma), and CPU affinity mask
#      - Uses srun only if SLURM_NTASKS > 1
# ================================================================

set -euo pipefail
module load rocm/6.4
# If compilers aren't on PATH, uncomment as needed:
# module load gcc

THREAD_COUNTS=(1 2 4)

echo "=== RUN START $(date) on $(hostname) ==="
echo "SLURM: JOB_ID=${SLURM_JOB_ID:-?} NODES=${SLURM_JOB_NUM_NODES:-?} NTASKS=${SLURM_NTASKS:-?} TPERNODE=${SLURM_NTASKS_PER_NODE:-?} CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

# Decide whether to use srun for Experiments B and C
USE_SRUN_FOR_BC=0
if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
  USE_SRUN_FOR_BC=1
fi
echo "[INFO] USE_SRUN_FOR_BC=${USE_SRUN_FOR_BC} (1 => B and C will use srun when >1 tasks allocated)"

# -------------------------------
# Basic system info (safe subset)
# -------------------------------
echo
echo "========== SLURM VERSION =========="
scontrol --version || true
srun --version || true
sinfo --version || true

echo
echo "========== SLURM FULL JOB SETTINGS =========="
if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol show job -dd "${SLURM_JOB_ID}" || scontrol show job "${SLURM_JOB_ID}" || true
  # Safer step query (avoid wildcard which may segfault)
  scontrol show step "${SLURM_JOB_ID}.batch" || true
fi

echo
echo "========== SOFTWARE VERSIONS (HIP, ROCm) =========="
if command -v rocm-smi >/dev/null 2>&1; then
  # Use only supported flags on your rocm-smi
  rocm-smi --showdriverversion --showproductname --showbus || true
fi
if command -v hipcc >/dev/null 2>&1; then hipcc --version || true; fi
if command -v hipconfig >/dev/null 2>&1; then
  hipconfig --version || true
  hipconfig --platforms || true
fi

echo
echo "========== SYSTEM TOPOLOGY =========="
if command -v lscpu >/dev/null 2>&1; then
  lscpu | egrep 'Architecture|Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)'
fi
if command -v lstopo-no-graphics >/dev/null 2>&1; then
  echo "-- HWLOC (NUMA only) --"
  lstopo-no-graphics --only numa || true
fi
echo "====================================="

# -------------------------------------------------------
# EXPERIMENT A: hello_jobstep-style (HIP + OpenMP, srun)
#   - Dynamically compute RT_GPU_ID, Bus_ID, GPU_ID; per-thread NUMA via optional libnuma
# -------------------------------------------------------
HELLO_STEP_SRC="/tmp/hello_jobstep_gpu_${SLURM_JOB_ID}_$$.cpp"
HELLO_STEP_BIN="/tmp/hello_jobstep_gpu_${SLURM_JOB_ID}_$$"

cat > "${HELLO_STEP_SRC}" << 'EOF'
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <limits.h>
#include <omp.h>
#include <dlfcn.h>

// Optional NUMA via dlopen
typedef int (*numa_available_fn)(void);
typedef int (*numa_node_of_cpu_fn)(int);

static void* numa_handle = nullptr;
static numa_available_fn p_numa_available = nullptr;
static numa_node_of_cpu_fn p_numa_node_of_cpu = nullptr;

static void init_numa() {
  // Try common libnuma sonames
  const char* candidates[] = {"libnuma.so.1", "libnuma.so", nullptr};
  for (int i = 0; candidates[i]; ++i) {
    numa_handle = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
    if (numa_handle) break;
  }
  if (numa_handle) {
    p_numa_available = (numa_available_fn)dlsym(numa_handle, "numa_available");
    p_numa_node_of_cpu = (numa_node_of_cpu_fn)dlsym(numa_handle, "numa_node_of_cpu");
  }
}

static int get_numa_node_for_cpu(int cpu) {
  if (!p_numa_available || !p_numa_node_of_cpu) return -1;
  int avail = p_numa_available();
  if (avail == -1) return -1;
  int nn = p_numa_node_of_cpu(cpu);
  return (nn >= 0) ? nn : -1;
}

// Lowercase helper
static void tolower_str(char* s) {
  for (; *s; ++s) if (*s >= 'A' && *s <= 'Z') *s = char(*s - 'A' + 'a');
}

// Extract 2-hex-digit bus from PCI BDF string "0000:bb:dd.f"
static void bus_hex_from_bdf(const char* bdf, char out[8]) {
  out[0] = 0;
  if (!bdf) return;
  const char* p = strchr(bdf, ':'); if (!p) return; p++;     // after domain
  const char* q = strchr(p, ':');  if (!q) return;            // end of bus
  size_t n = size_t(q - p);
  if (n > 2) n = 2;
  strncpy(out, p, n);
  out[n] = 0;
  tolower_str(out);
}

// Map bus hex (e.g., "d1") to DRM cardN by scanning /sys/class/drm/card*/device
static int physical_gpu_id_from_bus(const char* bus_hex) {
  DIR* d = opendir("/sys/class/drm");
  if (!d) return -1;
  struct dirent* de;
  int result = -1;
  while ((de = readdir(d)) != nullptr) {
    if (strncmp(de->d_name, "card", 4) != 0) continue;
    char devlink[256];
    snprintf(devlink, sizeof(devlink), "/sys/class/drm/%s/device", de->d_name);
    char buf[PATH_MAX];
    ssize_t n = readlink(devlink, buf, sizeof(buf)-1);
    if (n <= 0) continue;
    buf[n] = 0; // readlink doesn't null-terminate
    const char* p = strstr(buf, "0000:");
    if (!p) continue;
    char bhex[8]; bhex[0] = 0;
    bus_hex_from_bdf(p, bhex);
    if (strcmp(bhex, bus_hex) == 0) {
      int phys = -1;
      if (sscanf(de->d_name, "card%d", &phys) == 1) {
        result = phys;
        break;
      }
    }
  }
  closedir(d);
  return result;
}

int main() {
  init_numa();

  // Rank and node info
  int mpi_rank = 0;
  if (const char* s = getenv("SLURM_PROCID")) mpi_rank = atoi(s);
  char node[256];
  gethostname(node, sizeof(node)); node[sizeof(node)-1] = 0;

  // HIP device discovery (respects HIP_VISIBLE_DEVICES)
  int deviceCount = 0;
  hipError_t e = hipGetDeviceCount(&deviceCount);
  int rt_gpu_id = -1;
  char bdf[64] = {0};
  char bus_hex[8] = {0};
  int phys_gpu_id = -1;

  if (e == hipSuccess && deviceCount > 0) {
    (void)hipSetDevice(0);
    (void)hipGetDevice(&rt_gpu_id);

    // Bus ID
    if (hipDeviceGetPCIBusId(bdf, sizeof(bdf), rt_gpu_id) == hipSuccess) {
      bus_hex_from_bdf(bdf, bus_hex);                 // "d1"
      phys_gpu_id = physical_gpu_id_from_bus(bus_hex); // cardN -> N
    }
  }

  // hello_jobstep-style: per OMP thread print
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int hwt = sched_getcpu();
    int numa_node = (hwt >= 0) ? get_numa_node_for_cpu(hwt) : -1;
    #pragma omp critical
    {
      printf("A: MPI %03d - OMP %03d - HWT %03d - NUMA %d - Node %s - RT_GPU_ID %d - GPU_ID %d - Bus_ID %s\n",
             mpi_rank, tid, hwt, numa_node, node,
             rt_gpu_id, phys_gpu_id, (bus_hex[0]?bus_hex:"--"));
    }
  }

  return 0;
}
EOF

echo "[INFO] Compiling hello_jobstep (HIP + OpenMP, NUMA optional via dlopen)..."
hipcc -O2 -std=c++17 -fopenmp -o "${HELLO_STEP_BIN}" "${HELLO_STEP_SRC}"
echo "✓ Built: ${HELLO_STEP_BIN}"

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
for T in "${THREAD_COUNTS[@]}"; do
  echo
  echo "========== EXPERIMENT A: hello_jobstep (HIP+OMP) OMP_NUM_THREADS=${T} =========="
  OMP_NUM_THREADS="${T}" srun --export=ALL "${HELLO_STEP_BIN}" | sort
done

# -------------------------------------------------------
# EXPERIMENT B: POSIX pthreads affinity (conditionally srun)
#   - Pure pthreads (no OpenMP), prints per-thread CPU mask, HWT, NUMA (optional via dlopen)
# -------------------------------------------------------
CPU_AFF_SRC="/tmp/pthreads_affinity_${SLURM_JOB_ID}_$$.c"
CPU_AFF_BIN="/tmp/pthreads_affinity_${SLURM_JOB_ID}_$$"

cat > "${CPU_AFF_SRC}" << 'EOF'
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

typedef int (*numa_available_fn)(void);
typedef int (*numa_node_of_cpu_fn)(int);

static void* numa_handle = NULL;
static numa_available_fn p_numa_available = NULL;
static numa_node_of_cpu_fn p_numa_node_of_cpu = NULL;

static void init_numa() {
    const char* cands[] = {"libnuma.so.1", "libnuma.so", NULL};
    for (int i = 0; cands[i]; ++i) {
        numa_handle = dlopen(cands[i], RTLD_LAZY | RTLD_LOCAL);
        if (numa_handle) break;
    }
    if (numa_handle) {
        p_numa_available = (numa_available_fn)dlsym(numa_handle, "numa_available");
        p_numa_node_of_cpu = (numa_node_of_cpu_fn)dlsym(numa_handle, "numa_node_of_cpu");
    }
}

static int get_numa_node_for_cpu(int cpu) {
    if (!p_numa_available || !p_numa_node_of_cpu) return -1;
    int avail = p_numa_available();
    if (avail == -1) return -1;
    int nn = p_numa_node_of_cpu(cpu);
    return (nn >= 0) ? nn : -1;
}

typedef struct { int tid; } arg_t;
static pthread_mutex_t print_lock = PTHREAD_MUTEX_INITIALIZER;

void* worker(void* vp) {
    arg_t* a = (arg_t*)vp;
    cpu_set_t cpuset;
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_getaffinity_np");
        return NULL;
    }
    int hwt = sched_getcpu();
    int numa_node = (hwt >= 0) ? get_numa_node_for_cpu(hwt) : -1;

    char node[256]; gethostname(node, sizeof(node)); node[sizeof(node)-1] = 0;

    pthread_mutex_lock(&print_lock);
    printf("B: Node %s - PTHREAD %03d - HWT %03d - NUMA %d - Affinity: ", node, a->tid, hwt, numa_node);
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &cpuset)) printf("%d ", i);
    printf("\n");
    pthread_mutex_unlock(&print_lock);
    return NULL;
}

int main() {
    init_numa();

    int nthreads = 1;
    const char* p = getenv("THREADS");
    if (p) {
        int v = atoi(p);
        if (v > 0) nthreads = v;
    }

    printf("[B:Context] PID=%d, THREADS=%d\n", getpid(), nthreads);

    pthread_t* th = (pthread_t*)calloc(nthreads, sizeof(pthread_t));
    arg_t* args = (arg_t*)calloc(nthreads, sizeof(arg_t));
    if (!th || !args) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    for (int i = 0; i < nthreads; ++i) {
        args[i].tid = i;
        if (pthread_create(&th[i], NULL, worker, &args[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
    }
    for (int i = 0; i < nthreads; ++i) pthread_join(th[i], NULL);
    free(th); free(args);
    return 0;
}
EOF

echo "[INFO] Compiling pthreads_affinity (NUMA optional via dlopen)..."
gcc -O2 -std=c11 -pthread -o "${CPU_AFF_BIN}" "${CPU_AFF_SRC}"
echo "✓ Built: ${CPU_AFF_BIN}"

for T in "${THREAD_COUNTS[@]}"; do
  echo
  echo "========== EXPERIMENT B: POSIX pthreads, THREADS=${T} =========="
  if [ "${USE_SRUN_FOR_BC}" -eq 1 ]; then
    THREADS="${T}" srun --export=ALL "${CPU_AFF_BIN}" | sort
  else
    THREADS="${T}" "${CPU_AFF_BIN}"
  fi
done

# -------------------------------------------------------
# EXPERIMENT C: OpenMP thread affinity (conditionally srun)
#   - Prints per-thread CPU mask, HWT, NUMA (optional via dlopen)
# -------------------------------------------------------
OMP_AFF_SRC="/tmp/omp_affinity_${SLURM_JOB_ID}_$$.c"
OMP_AFF_BIN="/tmp/omp_affinity_${SLURM_JOB_ID}_$$"

cat > "${OMP_AFF_SRC}" << 'EOF'
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <omp.h>
#include <dlfcn.h>

typedef int (*numa_available_fn)(void);
typedef int (*numa_node_of_cpu_fn)(int);

static void* numa_handle = NULL;
static numa_available_fn p_numa_available = NULL;
static numa_node_of_cpu_fn p_numa_node_of_cpu = NULL;

static void init_numa() {
    const char* cands[] = {"libnuma.so.1", "libnuma.so", NULL};
    for (int i = 0; cands[i]; ++i) {
        numa_handle = dlopen(cands[i], RTLD_LAZY | RTLD_LOCAL);
        if (numa_handle) break;
    }
    if (numa_handle) {
        p_numa_available = (numa_available_fn)dlsym(numa_handle, "numa_available");
        p_numa_node_of_cpu = (numa_node_of_cpu_fn)dlsym(numa_handle, "numa_node_of_cpu");
    }
}

static int get_numa_node_for_cpu(int cpu) {
    if (!p_numa_available || !p_numa_node_of_cpu) return -1;
    int avail = p_numa_available();
    if (avail == -1) return -1;
    int nn = p_numa_node_of_cpu(cpu);
    return (nn >= 0) ? nn : -1;
}

static void print_mask(const cpu_set_t* cs) {
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, cs)) printf("%d ", i);
}

int main() {
    init_numa();

    char node[256]; gethostname(node, sizeof(node)); node[sizeof(node)-1] = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            #pragma omp critical
            perror("pthread_getaffinity_np");
        }
        int hwt = sched_getcpu();
        int numa_node = (hwt >= 0) ? get_numa_node_for_cpu(hwt) : -1;

        #pragma omp critical
        {
            printf("C: Node %s - OMP %03d - HWT %03d - NUMA %d - Affinity: ", node, tid, hwt, numa_node);
            print_mask(&cpuset);
            printf("\n");
        }
    }
    return 0;
}
EOF

echo "[INFO] Compiling omp_affinity (NUMA optional via dlopen)..."
gcc -O2 -std=c11 -fopenmp -pthread -o "${OMP_AFF_BIN}" "${OMP_AFF_SRC}"
echo "✓ Built: ${OMP_AFF_BIN}"

for T in "${THREAD_COUNTS[@]}"; do
  echo
  echo "========== EXPERIMENT C: OpenMP, OMP_NUM_THREADS=${T} =========="
  if [ "${USE_SRUN_FOR_BC}" -eq 1 ]; then
    OMP_NUM_THREADS="${T}" srun --export=ALL "${OMP_AFF_BIN}" | sort
  else
    OMP_NUM_THREADS="${T}" "${OMP_AFF_BIN}"
  fi
done

# Cleanup
rm -f "${HELLO_STEP_SRC}" "${HELLO_STEP_BIN}" \
      "${CPU_AFF_SRC}" "${CPU_AFF_BIN}" \
      "${OMP_AFF_SRC}" "${OMP_AFF_BIN}"

echo "=== RUN END $(date) ==="

