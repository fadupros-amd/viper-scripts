#!/bin/bash
#SBATCH -J Affinity
#SBATCH -o binding_%j.out
#SBATCH -e binding_%j.err
#SBATCH --ntasks-per-node=1   # Match your desired ranks per node
#SBATCH --nodes=1             # Match your Trainer(num_nodes=...)
#SBATCH --cpus-per-task=24
#SBATCH -p MI300A_A1_COS_OK
#SBATCH -t 0:05:00
#SBATCH --exclusive

### Script generated with the Help of NABU AI


# ================================================================
# Experiments:
#   A) Unbound info-only (srun; single-threaded) - uses job allocation
#   B) CPU affinity probe (no srun; POSIX pthreads only) - exactly 1 thread
#   C) OpenMP thread affinity probe (no srun; OpenMP) - exactly 1 thread
#
# Also prints:
#   - Full Slurm job settings (scontrol show job -dd and step info)
#   - Hyperthreading detection via lscpu
#   - NUMA topology via hwloc (if installed)
#   - HIP/ROCm versions
# ROCm: 6.4
# ================================================================

set -euo pipefail
module load rocm/6.4
# If gcc isn't available by default, uncomment:
# module load gcc

echo "=== RUN START $(date) on $(hostname) ==="
echo "SLURM: NODES=${SLURM_JOB_NUM_NODES:-?} NTASKS=${SLURM_NTASKS:-?} TPERNODE=${SLURM_NTASKS_PER_NODE:-?} CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

# -------------------------------------------------------
# SLURM VERSION AND CONFIG SNAPSHOT
# -------------------------------------------------------
echo
echo "========== SLURM VERSION =========="
scontrol --version || true
srun --version || true
sinfo --version || true

echo
echo "========== SLURM FULL JOB SETTINGS =========="
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "-- scontrol show job -dd ${SLURM_JOB_ID} --"
  scontrol show job -dd "${SLURM_JOB_ID}" || scontrol show job "${SLURM_JOB_ID}" || true
  echo
  echo "-- scontrol show step -dd ${SLURM_JOB_ID}.* (steps appear after launch) --"
  scontrol show step -dd "${SLURM_JOB_ID}.*" || true
else
  echo "SLURM_JOB_ID is not set"
fi

# -------------------------------------------------------
# SOFTWARE VERSIONS: HIP, ROCm
# -------------------------------------------------------
echo
echo "========== SOFTWARE VERSIONS (HIP, ROCm) =========="
if command -v rocm-smi >/dev/null 2>&1; then
  echo "-- rocm-smi --showdriverversion --showversion --"
  rocm-smi --showdriverversion --showversion || true
fi
if command -v hipcc >/dev/null 2>&1; then
  echo "-- hipcc --version --"
  hipcc --version || true
fi
if command -v hipconfig >/dev/null 2>&1; then
  echo "-- hipconfig --version --"
  hipconfig --version || true
  echo "-- hipconfig --platforms --"
  hipconfig --platforms || true
fi

# -------------------------------
# System topology summary (lscpu + hwloc)
# -------------------------------
echo
echo "========== SYSTEM TOPOLOGY =========="
if command -v lscpu >/dev/null 2>&1; then
  TPC=$(lscpu | awk -F: '/Thread\(s\) per core/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
  CPC=$(lscpu | awk -F: '/Core\(s\) per socket/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
  SCK=$(lscpu | awk -F: '/Socket\(s\)/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
  ONODES=$(lscpu | awk -F: '/NUMA node\(s\)/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
  HT_STATUS="disabled"
  if [ -n "${TPC}" ] && [ "${TPC}" -gt 1 ] 2>/dev/null; then
    HT_STATUS="enabled (threads per core = ${TPC})"
  elif [ -n "${TPC}" ]; then
    HT_STATUS="disabled (threads per core = ${TPC})"
  fi
  echo "Hyperthreading: ${HT_STATUS}"
  echo "Sockets: ${SCK:-?}, Cores/socket: ${CPC:-?}, NUMA nodes: ${ONODES:-?}"
  echo "-- lscpu key fields --"
  lscpu | egrep 'Architecture|Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)'
else
  echo "lscpu not found"
fi

echo
if command -v lstopo-no-graphics >/dev/null 2>&1; then
  echo "========== HWLOC NUMA TOPOLOGY (lstopo-no-graphics) =========="
  lstopo-no-graphics --only numa || true
  echo "-- full topology --"
  lstopo-no-graphics || true
elif command -v hwloc-ls >/dev/null 2>&1; then
  echo "========== HWLOC NUMA TOPOLOGY (hwloc-ls) =========="
  hwloc-ls --only numa || true
  echo "-- full topology --"
  hwloc-ls || true
else
  echo "hwloc tools not available (lstopo-no-graphics/hwloc-ls)"
fi
echo "====================================="
echo

# -------------------------------
# Sanity: ensure >= 2 gfx942 GPUs exist (for Experiment A)
# -------------------------------
MI300A_GPUS=()
GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "^GPU\[") || GPU_COUNT=0
for i in $(seq 0 $((GPU_COUNT - 1))); do
  GFX=$(rocm-smi --showproductname -d "$i" 2>/dev/null | awk '/GFX Version/{print $NF}')
  [ "$GFX" = "gfx942" ] && MI300A_GPUS+=("$i")
done
if [ "${#MI300A_GPUS[@]}" -lt 2 ]; then
  echo "ERROR: Expected at least 2 gfx942 devices, found ${#MI300A_GPUS[@]} of $GPU_COUNT total"
  rocm-smi --showproductname || true
  exit 1
fi
echo "[INFO] GPUs detected (gfx942): ${MI300A_GPUS[*]}"

# -------------------------------------------------------
# Build HIP hello_jobstep tool (Experiment A)
# -------------------------------------------------------
HELLO_SRC="/tmp/hello_jobstep_hip_${SLURM_JOB_ID}_$$.cpp"
HELLO_BIN="/tmp/hello_jobstep_hip_${SLURM_JOB_ID}_$$"

cat > "${HELLO_SRC}" << 'ENDCPP'
#include <hip/hip_runtime.h>
#include <sched.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <dirent.h>
#include <limits.h>

static int get_cpu() { int c = sched_getcpu(); return c < 0 ? -1 : c; }

static void tolower_str(char* s){ for(;*s;++s){ if(*s>='A' && *s<='Z') *s = *s - 'A' + 'a'; } }

static void bus_hex_from_bdf(const char* bdf, char out[8]) {
  out[0]=0;
  if (!bdf) return;
  const char* p = strchr(bdf, ':'); if (!p) return; p++;
  const char* q = strchr(p, ':');  if (!q) return;
  size_t n = (size_t)(q - p);
  if (n > 2) n = 2;
  strncpy(out, p, n);
  out[n] = 0;
  tolower_str(out);
}

static int physical_gpu_id_from_bus(const char* bus_hex) {
  DIR* d = opendir("/sys/class/drm");
  if (!d) return -1;
  struct dirent* de;
  int result = -1;
  while ((de = readdir(d)) != NULL) {
    if (strncmp(de->d_name, "card", 4) != 0) continue;
    char devlink[256];
    snprintf(devlink, sizeof(devlink), "/sys/class/drm/%s/device", de->d_name);
    char buf[PATH_MAX];
    ssize_t n = readlink(devlink, buf, sizeof(buf)-1);
    if (n <= 0) continue;
    buf[n] = 0;
    const char* p = strstr(buf, "0000:");
    if (!p) continue;
    char bhex[8]; bhex[0]=0;
    bus_hex_from_bdf(p, bhex);
    if (strcmp(bhex, bus_hex) == 0) {
      int phys = -1;
      if (sscanf(de->d_name, "card%d", &phys) == 1) { result = phys; break; }
    }
  }
  closedir(d);
  return result;
}

int main() {
  int mpi_rank = 0;
  if (const char* s = getenv("SLURM_PROCID")) mpi_rank = atoi(s);

  int omp_tid = 0;
#ifdef _OPENMP
  omp_tid = omp_get_thread_num();
#endif

  char node[256];
  gethostname(node, sizeof(node)); node[sizeof(node)-1]=0;

  int deviceCount = 0;
  (void)hipGetDeviceCount(&deviceCount);

  int rt_gpu_id = -1;
  if (deviceCount > 0) {
    (void)hipSetDevice(0); // respects HIP_VISIBLE_DEVICES
    (void)hipGetDevice(&rt_gpu_id);
  }

  char bus_hex[8]; bus_hex[0]=0;
  int phys_gpu_id = -1;
  if (rt_gpu_id >= 0) {
    char bdf[64] = {0};
    (void)hipDeviceGetPCIBusId(bdf, sizeof(bdf), rt_gpu_id);
    bus_hex_from_bdf(bdf, bus_hex);
    phys_gpu_id = physical_gpu_id_from_bus(bus_hex);
  }

  int hwt = get_cpu();

  printf("MPI %03d - OMP %03d - HWT %03d - Node %s - RT_GPU_ID %d - GPU_ID %d - Bus_ID %s\n",
         mpi_rank, omp_tid, hwt, node, rt_gpu_id, phys_gpu_id, (bus_hex[0]?bus_hex:"--"));
  return 0;
}
ENDCPP

echo "[INFO] Compiling hello_jobstep_hip..."
hipcc -O2 -fopenmp -o "${HELLO_BIN}" "${HELLO_SRC}"
echo "✓ Built: ${HELLO_BIN}"

# -------------------------------------------------------
# EXPERIMENT B: CPU affinity probe (no srun; POSIX pthreads ONLY, single thread)
# -------------------------------------------------------
CPU_AFF_SRC="/tmp/cpu_affinity_threads_${SLURM_JOB_ID}_$$.c"
CPU_AFF_BIN="/tmp/cpu_affinity_threads_${SLURM_JOB_ID}_$$"

cat > "${CPU_AFF_SRC}" << 'EOF'
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

static void print_allowed_lists(void) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Cpus_allowed_list:", 18)==0 ||
            strncmp(line, "Mems_allowed_list:", 18)==0) {
            fprintf(stdout, "%s", line);
        }
    }
    fclose(f);
}

void* worker(void* vp) {
    cpu_set_t cpuset;
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_getaffinity_np");
        return NULL;
    }
    int hwt = sched_getcpu();
    char node[256]; gethostname(node, sizeof(node)); node[sizeof(node)-1] = 0;

    printf("Node %s - POSIX_TID %03d - HWT %03d - Affinity: ", node, 0, hwt);
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &cpuset)) printf("%d ", i);
    printf("\n");
    return NULL;
}

int main() {
    const int nthreads = 1; // exactly one thread as requested
    printf("[POSIX] PID=%d, Threads=%d\n", getpid(), nthreads);
    print_allowed_lists();

    pthread_t th;
    if (pthread_create(&th, NULL, worker, NULL) != 0) {
        perror("pthread_create");
        return 1;
    }
    pthread_join(th, NULL);
    return 0;
}
EOF

echo "[INFO] Compiling cpu_affinity (POSIX pthreads)..."
gcc -O2 -std=c11 -pthread -o "${CPU_AFF_BIN}" "${CPU_AFF_SRC}"
echo "✓ Built: ${CPU_AFF_BIN}"

# -------------------------------------------------------
# EXPERIMENT C: OpenMP thread affinity probe (no srun; exactly 1 OpenMP thread)
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
#include <string.h>

static void print_allowed_lists(void) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Cpus_allowed_list:", 18)==0 ||
            strncmp(line, "Mems_allowed_list:", 18)==0) {
            fprintf(stdout, "%s", line);
        }
    }
    fclose(f);
}

static void print_mask(const cpu_set_t* cs) {
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, cs)) printf("%d ", i);
}

int main() {
    char node[256]; gethostname(node, sizeof(node)); node[sizeof(node)-1] = 0;

    int omp_threads = 1;
    printf("[OpenMP] PID=%d, OMP_NUM_THREADS=%d\n", getpid(), omp_threads);
    print_allowed_lists();

    omp_set_num_threads(1); // force exactly 1 OMP thread
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            #pragma omp critical
            perror("pthread_getaffinity_np");
        }
        int hwt = sched_getcpu();
        #pragma omp critical
        {
            printf("Node %s - OMP %03d - HWT %03d - Affinity: ", node, tid, hwt);
            print_mask(&cpuset);
            printf("\n");
        }
    }
    return 0;
}
EOF

echo "[INFO] Compiling omp_affinity (OpenMP)..."
gcc -O2 -std=c11 -fopenmp -pthread -o "${OMP_AFF_BIN}" "${OMP_AFF_SRC}"
echo "✓ Built: ${OMP_AFF_BIN}"

# -------------------------------------------------------
# EXPERIMENT A: Unbound info-only (srun; single-threaded)
#   - Purpose: observe default placement for a single task and single thread
#   - Notes: program is single-threaded; set OMP_NUM_THREADS=1 for consistency
# -------------------------------------------------------
echo
echo "========== EXPERIMENT A: Unbound info-only (srun, OMP_NUM_THREADS=1) =========="
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
echo "[A:Context] Launching with srun (no explicit --ntasks/--cpus-per-task)"
echo "[A:Env] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[A:Proc pre-run] PID=$$, Cpus_allowed_list=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status), Mems_allowed_list=$(awk '/Mems_allowed_list/{print $2}' /proc/self/status)"
srun --export=ALL "${HELLO_BIN}" | sort
echo "[A:Post] For step info, see: scontrol show step ${SLURM_JOB_ID}.*"

# -------------------------------------------------------
# EXPERIMENT B: CPU affinity probe (no srun; POSIX pthreads only, 1 thread)
#   - Purpose: see CPU mask and HWT for a single POSIX thread
# -------------------------------------------------------
echo
echo "========== EXPERIMENT B: POSIX pthreads, Threads=1 =========="
echo "[B:Proc pre-run] PID=$$, Cpus_allowed_list=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status), Mems_allowed_list=$(awk '/Mems_allowed_list/{print $2}' /proc/self/status)"
"${CPU_AFF_BIN}"

# -------------------------------------------------------
# EXPERIMENT C: OpenMP thread affinity probe (no srun; OMP_NUM_THREADS=1)
#   - Purpose: see CPU mask and HWT for a single OpenMP thread
# -------------------------------------------------------
echo
echo "========== EXPERIMENT C: OpenMP, OMP_NUM_THREADS=1 =========="
export OMP_NUM_THREADS=1
echo "[C:Env] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[C:Proc pre-run] PID=$$, Cpus_allowed_list=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status), Mems_allowed_list=$(awk '/Mems_allowed_list/{print $2}' /proc/self/status)"
"${OMP_AFF_BIN}"

# Cleanup
rm -f "${HELLO_SRC}" "${HELLO_BIN}" \
      "${CPU_AFF_SRC}" "${CPU_AFF_BIN}" \
      "${OMP_AFF_SRC}" "${OMP_AFF_BIN}"

echo "=== RUN END $(date) ==="

