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


# Create the C source file
cat > cpu_affinity.c << 'EOF'
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>

int main() {
    pthread_t self_thread = pthread_self();
    cpu_set_t cpuset;
    int s;

    // Get the current CPU affinity mask of the thread
    s = pthread_getaffinity_np(self_thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) {
        perror("pthread_getaffinity_np");
        exit(EXIT_FAILURE);
    }

    printf("Thread %lu affinity mask: ", self_thread);
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset)) {
            printf("%d ", i);
        }
    }
    printf("\n");

    return 0;
}
EOF

# Compile the program
echo "Compiling cpu_affinity.c..."
gcc -o cpu_affinity cpu_affinity.c -pthread

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running the program..."
    echo "------------------------"
    srun ./cpu_affinity
    echo "------------------------"
else
    echo "Compilation failed!"
    exit 1
fi

