#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

float maxRuleDistance;

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_posSorted;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < N)
    {
        glm::vec3 rand = generateRandomVec3(time, index);
        arr[index].x = scale * rand.x;
        arr[index].y = scale * rand.y;
        arr[index].z = scale * rand.z;
    }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid(utilityCore::divup(N, blockSize));

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_posSorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_posSorted failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");
  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  maxRuleDistance = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  gridCellWidth = 2.0f * maxRuleDistance;
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid(utilityCore::divup(numObjects, blockSize));

  kernCopyPositionsToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel)
{
    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    // Rule 2: boids try to stay a distance d away from each other
    // Rule 3: boids try to match the speed of surrounding boids
    float rule1Neighbors = 0;
    glm::vec3 perceivedCenter(0.0f, 0.0f, 0.0f);

    glm::vec3 avoidVel(0.0f, 0.0f, 0.0f);

    float rule3Neighbors = 0;
    glm::vec3 perceivedVelocity(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < N; i++)
    {
        if (i == iSelf)
        {
            continue;
        }

        glm::vec3 curToNeighbor = pos[i] - pos[iSelf];
        float distToNeighbor = glm::length(curToNeighbor);

        if (distToNeighbor < rule1Distance)
        {
            perceivedCenter += pos[i];
            rule1Neighbors++;
        }

        if (distToNeighbor < rule2Distance)
        {
            avoidVel -= curToNeighbor;
        }

        if (distToNeighbor < rule3Distance)
        {
            perceivedVelocity += vel[i];
            rule3Neighbors++;
        }
    }

    glm::vec3 rule1Vel = rule1Neighbors
        ? (((perceivedCenter / rule1Neighbors) - pos[iSelf]) * rule1Scale)
        : glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 rule3Vel = rule3Neighbors
        ? ((perceivedVelocity / rule3Neighbors) * rule3Scale)
        : glm::vec3(0.0f, 0.0f, 0.0f);

    return rule1Vel
        + ((avoidVel) * rule2Scale)
        + rule3Vel;
}

/**
* TODO-1.1 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
    glm::vec3 *vel1, glm::vec3 *vel2)
{
    // Compute a new velocity based on pos and vel1
    // Clamp the speed
    // Record the new velocity into vel2. Question: why NOT vel1?
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= N)
    {
        return;
    }

    glm::vec3 newVel = vel1[idx] + computeVelocityChange(N, idx, pos, vel1);

    if (float speed = glm::length(newVel); speed > maxSpeed)
    {
        newVel = (newVel / speed) * maxSpeed;
    }

    vel2[idx] = newVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution)
{
    return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices)
{
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int boidIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (boidIdx >= N)
    {
        return;
    }

    indices[boidIdx] = boidIdx;

    glm::vec3 boidPos = pos[boidIdx];
    int iX = floor((boidPos.x - gridMin.x) * inverseCellWidth);
    int iY = floor((boidPos.y - gridMin.y) * inverseCellWidth);
    int iZ = floor((boidPos.z - gridMin.z) * inverseCellWidth);
    int cellIdx = gridIndex3Dto1D(iX, iY, iZ, gridResolution);
    gridIndices[boidIdx] = cellIdx;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N)
    {
        return;
    }

    intBuffer[index] = value;
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices)
{
    // TODO-2.1
    // Identify the start point of each cell in the gridIndices array.
    // This is basically a parallel unrolling of a loop that goes
    // "this index doesn't match the one before it, must be a new cell!"
    int sortedBoidIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (sortedBoidIdx >= N)
    {
        return;
    }
        
    int cellIdx = particleGridIndices[sortedBoidIdx];
    bool isCellStart = sortedBoidIdx == 0 || particleGridIndices[sortedBoidIdx - 1] != cellIdx;
    bool isCellEnd = sortedBoidIdx == N - 1 || particleGridIndices[sortedBoidIdx + 1] != cellIdx;

    if (isCellStart)
    {
        gridCellStartIndices[cellIdx] = sortedBoidIdx;
    }

    if (isCellEnd)
    {
        gridCellEndIndices[cellIdx] = sortedBoidIdx;
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  float maxRuleDist, int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2)
{
    // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked.
    // - Identify the grid cell that this particle is in
    // - Identify which cells may contain neighbors. This isn't always 8.
    // - For each cell, read the start/end indices in the boid pointer array.
    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    // - Clamp the speed change before putting the new speed in vel2
    int sortedBoidIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (sortedBoidIdx >= N)
    {
        return;
    }

    int unsortedBoidIdx = particleArrayIndices[sortedBoidIdx];
    glm::vec3 boidPos = pos[unsortedBoidIdx];
    
    float minSearchPosX = boidPos.x - maxRuleDist;
    float maxSearchPosX = boidPos.x + maxRuleDist;
    float minSearchPosY = boidPos.y - maxRuleDist;
    float maxSearchPosY = boidPos.y + maxRuleDist;
    float minSearchPosZ = boidPos.z - maxRuleDist;
    float maxSearchPosZ = boidPos.z + maxRuleDist;

    int minXCellIdx = floor((minSearchPosX - gridMin.x) * inverseCellWidth);
    int maxXCellIdx = floor((maxSearchPosX - gridMin.x) * inverseCellWidth);
    int minYCellIdx = floor((minSearchPosY - gridMin.y) * inverseCellWidth);
    int maxYCellIdx = floor((maxSearchPosY - gridMin.y) * inverseCellWidth);
    int minZCellIdx = floor((minSearchPosZ - gridMin.z) * inverseCellWidth);
    int maxZCellIdx = floor((maxSearchPosZ - gridMin.z) * inverseCellWidth);

    int maxCellSideIdx = gridResolution - 1;
    minXCellIdx = imax(imin(minXCellIdx, maxCellSideIdx), 0);
    maxXCellIdx = imax(imin(maxXCellIdx, maxCellSideIdx), 0);
    minYCellIdx = imax(imin(minYCellIdx, maxCellSideIdx), 0);
    maxYCellIdx = imax(imin(maxYCellIdx, maxCellSideIdx), 0);
    minZCellIdx = imax(imin(minZCellIdx, maxCellSideIdx), 0);
    maxZCellIdx = imax(imin(maxZCellIdx, maxCellSideIdx), 0);

    float rule1Neighbors = 0;
    glm::vec3 perceivedCenter(0.0f, 0.0f, 0.0f);

    glm::vec3 avoidVel(0.0f, 0.0f, 0.0f);

    float rule3Neighbors = 0;
    glm::vec3 perceivedVelocity(0.0f, 0.0f, 0.0f);

    for (int iZ = minZCellIdx; iZ <= maxZCellIdx; iZ++)
    {
        for (int iY = minYCellIdx; iY <= maxYCellIdx; iY++)
        {
            for (int iX = minXCellIdx; iX <= maxXCellIdx; iX++)
            {
                // convert 3D-->1D cellIdx
                int cellIdx = gridIndex3Dto1D(iX, iY, iZ, gridResolution);

                int gridCellStart = gridCellStartIndices[cellIdx];
                if (gridCellStart == -1)
                {
                    continue;
                }

                for (int sortedNeighborBoidIdx = gridCellStart; sortedNeighborBoidIdx <= gridCellEndIndices[cellIdx]; sortedNeighborBoidIdx++)
                {
                    if (sortedNeighborBoidIdx == sortedBoidIdx)
                    {
                        continue;
                    }

                    int unsortedNeighborBoidIdx = particleArrayIndices[sortedNeighborBoidIdx];
                    glm::vec3 neighborBoidPos = pos[unsortedNeighborBoidIdx];
                    glm::vec3 curToNeighbor = neighborBoidPos - boidPos;
                    float distToNeighbor = glm::length(curToNeighbor);

                    if (distToNeighbor < rule1Distance)
                    {
                        perceivedCenter += neighborBoidPos;
                        rule1Neighbors++;
                    }

                    if (distToNeighbor < rule2Distance)
                    {
                        avoidVel -= curToNeighbor;
                    }

                    if (distToNeighbor < rule3Distance)
                    {
                        perceivedVelocity += vel1[unsortedNeighborBoidIdx];
                        rule3Neighbors++;
                    }
                }
            }
        }
    }

    glm::vec3 rule1Vel = rule1Neighbors
        ? (((perceivedCenter / rule1Neighbors) - boidPos) * rule1Scale)
        : glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 rule3Vel = rule3Neighbors
        ? ((perceivedVelocity / rule3Neighbors) * rule3Scale)
        : glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 velChange = rule1Vel
        + ((avoidVel) * rule2Scale)
        + rule3Vel;

    glm::vec3 newVel = vel1[unsortedBoidIdx] + velChange;

    if (float speed = glm::length(newVel); speed > maxSpeed)
    {
        newVel = (newVel / speed) * maxSpeed;
    }

    vel2[unsortedBoidIdx] = newVel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  float maxRuleDist, int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2)
{
    // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
    // except with one less level of indirection.
    // This should expect gridCellStartIndices and gridCellEndIndices to refer
    // directly to pos and vel1.
    // - Identify the grid cell that this particle is in
    // - Identify which cells may contain neighbors. This isn't always 8.
    // - For each cell, read the start/end indices in the boid pointer array.
    //   DIFFERENCE: For best results, consider what order the cells should be
    //   checked in to maximize the memory benefits of reordering the boids data.
    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    // - Clamp the speed change before putting the new speed in vel2
    int sortedBoidIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (sortedBoidIdx >= N)
    {
        return;
    }

    glm::vec3 boidPos = pos[sortedBoidIdx];

    float minSearchPosX = boidPos.x - maxRuleDist;
    float maxSearchPosX = boidPos.x + maxRuleDist;
    float minSearchPosY = boidPos.y - maxRuleDist;
    float maxSearchPosY = boidPos.y + maxRuleDist;
    float minSearchPosZ = boidPos.z - maxRuleDist;
    float maxSearchPosZ = boidPos.z + maxRuleDist;

    int minXCellIdx = floor((minSearchPosX - gridMin.x) * inverseCellWidth);
    int maxXCellIdx = floor((maxSearchPosX - gridMin.x) * inverseCellWidth);
    int minYCellIdx = floor((minSearchPosY - gridMin.y) * inverseCellWidth);
    int maxYCellIdx = floor((maxSearchPosY - gridMin.y) * inverseCellWidth);
    int minZCellIdx = floor((minSearchPosZ - gridMin.z) * inverseCellWidth);
    int maxZCellIdx = floor((maxSearchPosZ - gridMin.z) * inverseCellWidth);

    // TODO: clamp all cell indices to [0, gridResolution - 1]
    int maxCellSideIdx = gridResolution - 1;
    minXCellIdx = imax(imin(minXCellIdx, maxCellSideIdx), 0);
    maxXCellIdx = imax(imin(maxXCellIdx, maxCellSideIdx), 0);
    minYCellIdx = imax(imin(minYCellIdx, maxCellSideIdx), 0);
    maxYCellIdx = imax(imin(maxYCellIdx, maxCellSideIdx), 0);
    minZCellIdx = imax(imin(minZCellIdx, maxCellSideIdx), 0);
    maxZCellIdx = imax(imin(maxZCellIdx, maxCellSideIdx), 0);

    float rule1Neighbors = 0;
    glm::vec3 perceivedCenter(0.0f, 0.0f, 0.0f);

    glm::vec3 avoidVel(0.0f, 0.0f, 0.0f);

    float rule3Neighbors = 0;
    glm::vec3 perceivedVelocity(0.0f, 0.0f, 0.0f);

    for (int iZ = minZCellIdx; iZ <= maxZCellIdx; iZ++)
    {
        for (int iY = minYCellIdx; iY <= maxYCellIdx; iY++)
        {
            for (int iX = minXCellIdx; iX <= maxXCellIdx; iX++)
            {
                // convert 3D-->1D cellIdx
                int cellIdx = gridIndex3Dto1D(iX, iY, iZ, gridResolution);

                int gridCellStart = gridCellStartIndices[cellIdx];
                if (gridCellStart == -1)
                {
                    continue;
                }

                for (int sortedNeighborBoidIdx = gridCellStart; sortedNeighborBoidIdx <= gridCellEndIndices[cellIdx]; sortedNeighborBoidIdx++)
                {
                    if (sortedNeighborBoidIdx == sortedBoidIdx)
                    {
                        continue;
                    }

                    glm::vec3 neighborBoidPos = pos[sortedNeighborBoidIdx];
                    glm::vec3 curToNeighbor = neighborBoidPos - boidPos;
                    float distToNeighbor = glm::length(curToNeighbor);

                    if (distToNeighbor < rule1Distance)
                    {
                        perceivedCenter += neighborBoidPos;
                        rule1Neighbors++;
                    }

                    if (distToNeighbor < rule2Distance)
                    {
                        avoidVel -= curToNeighbor;
                    }

                    if (distToNeighbor < rule3Distance)
                    {
                        perceivedVelocity += vel1[sortedNeighborBoidIdx];
                        rule3Neighbors++;
                    }
                }
            }
        }
    }

    glm::vec3 rule1Vel = rule1Neighbors
        ? (((perceivedCenter / rule1Neighbors) - boidPos) * rule1Scale)
        : glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 rule3Vel = rule3Neighbors
        ? ((perceivedVelocity / rule3Neighbors) * rule3Scale)
        : glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 velChange = rule1Vel
        + ((avoidVel)*rule2Scale)
        + rule3Vel;

    glm::vec3 newVel = vel1[sortedBoidIdx] + velChange;

    if (float speed = glm::length(newVel); speed > maxSpeed)
    {
        newVel = (newVel / speed) * maxSpeed;
    }

    vel2[sortedBoidIdx] = newVel;
}

__global__ void kernSetSortedBoidData(
    int N, int* particleArrayIndices, glm::vec3* pos,
    glm::vec3* sortedPos, glm::vec3* vel,
    glm::vec3* sortedVel)
{
    int sortedBoidIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (sortedBoidIdx >= N)
    {
        return;
    }

    int unsortedBoidIdx = particleArrayIndices[sortedBoidIdx];
    sortedPos[sortedBoidIdx] = pos[unsortedBoidIdx];
    sortedVel[sortedBoidIdx] = vel[unsortedBoidIdx];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt)
{
    // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    // TODO-1.2 ping-pong the velocity buffers
    dim3 fullBlocksPerGrid(utilityCore::divup(numObjects, blockSize));

    kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt)
{
    // TODO-2.1
    // Uniform Grid Neighbor search using Thrust sort.
    // In Parallel:
    // - label each particle with its array index as well as its grid index.
    //   Use 2x width grids.
    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    // - Perform velocity updates using neighbor search
    // - Update positions
    // - Ping-pong buffers as needed
    dim3 boidsFullBlocksPerGrid(utilityCore::divup(numObjects, blockSize));
    dim3 cellsFullBlocksPerGrid(utilityCore::divup(gridCellCount, blockSize));

    // Step 0: Use kernResetIntBuffer to set dev_gridCellStartIndices and dev_gridCellEndIndices
    // to -1 at each index. Spawn blocks covering cellCount threads.
    kernResetIntBuffer<<<cellsFullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer<<<cellsFullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);

    // Step 1: populate dev_particleGridIndices and dev_particleArrayIndices using kernComputeIndices.
    // Spawn blocks covering numObjects threads.
    kernComputeIndices<<<boidsFullBlocksPerGrid, blockSize>>>(
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices);

    // Step 2: use thrust::device_ptr and thrust::sort_by_key to sort both of these arrays on device
    // by the values stored in dev_particleGridIndices.
    thrust::sort_by_key(
        dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);

    // Step 3: use kernIdentifyCellStartEnd to populate dev_gridCellEndIndices and dev_gridCellStartIndices
    // by examining every index of dev_particleGridIndices. Spawn blocks covering numObjects threads.
    kernIdentifyCellStartEnd<<<boidsFullBlocksPerGrid, blockSize>>>(
        numObjects,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices);

    // Step 4: Use kernUpdateVelNeighborSearchScattered to compute the new velocity of each boid in a
    // given gridCell. A -1 value in dev_gridCellStartIndices / dev_gridCellEndIndices indicates an empty cell.
    // Spawn blocks covering numObjects threads.
    kernUpdateVelNeighborSearchScattered<<<boidsFullBlocksPerGrid, blockSize>>>(
        maxRuleDistance,
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_particleArrayIndices,
        dev_pos,
        dev_vel1,
        dev_vel2);

    // Step 5: Use kernUpdatePos to update all boid positions. Spawn blocks covering numObjects threads.
    kernUpdatePos<<<boidsFullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

    // Step 6: Ping-pong device velocity buffers
    glm::vec3* temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
}

void Boids::stepSimulationCoherentGrid(float dt)
{
    // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
    // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
    // In Parallel:
    // - Label each particle with its array index as well as its grid index.
    //   Use 2x width grids
    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
    //   the particle data in the simulation array.
    //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
    // - Perform velocity updates using neighbor search
    // - Update positions
    // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

    dim3 boidsFullBlocksPerGrid(utilityCore::divup(numObjects, blockSize));
    dim3 cellsFullBlocksPerGrid(utilityCore::divup(gridCellCount, blockSize));

    // Step 0: Use kernResetIntBuffer to set dev_gridCellStartIndices and dev_gridCellEndIndices
    // to -1 at each index. Spawn blocks covering cellCount threads.
    kernResetIntBuffer<<<cellsFullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer<<<cellsFullBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);

    // Step 1: populate dev_particleGridIndices and dev_particleArrayIndices using kernComputeIndices.
    // Spawn blocks covering numObjects threads.
    kernComputeIndices<<<boidsFullBlocksPerGrid, blockSize>>>(
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices);

    // Step 2: use thrust::device_ptr and thrust::sort_by_key to sort both of these arrays on device
    // by the values stored in dev_particleGridIndices.
    thrust::sort_by_key(
        dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);

    // Step 3a: Use kernIdentifyCellStartEnd to populate dev_gridCellEndIndices and dev_gridCellStartIndices
    // by examining every index of dev_particleGridIndices. Spawn blocks covering numObjects threads.
    kernIdentifyCellStartEnd<<<boidsFullBlocksPerGrid, blockSize>>>(
        numObjects,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices);

    // Step 3b: Use kernSetSortedBoidData to copy dev_pos and dev_vel1 into dev_posSorted and dev_vel1Sorted,
    // using dev_particleArrayIndices
    kernSetSortedBoidData<<<boidsFullBlocksPerGrid, blockSize>>>(
        numObjects,
        dev_particleArrayIndices,
        dev_pos,
        dev_posSorted,
        dev_vel1,
        dev_vel2);

    // Step 4: Use kernUpdateVelNeighborSearchCoherent to compute the new velocity of each boid in a
    // given gridCell. A -1 value in dev_gridCellStartIndices / dev_gridCellEndIndices indicates an empty cell.
    // Spawn blocks covering numObjects threads.
    kernUpdateVelNeighborSearchCoherent<<<boidsFullBlocksPerGrid, blockSize>>>(
        maxRuleDistance,
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_posSorted,
        dev_vel2,
        dev_vel1);

    // Step 5: Use kernUpdatePos to update all boid positions. Spawn blocks covering numObjects threads.
    kernUpdatePos<<<boidsFullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_posSorted, dev_vel1);

    // Step 6: Ping-pong device buffers
    glm::vec3 *temp = dev_pos;
    dev_pos = dev_posSorted;
    dev_posSorted = temp;
}

void Boids::endSimulation()
{
    cudaFree(dev_vel1);
    cudaFree(dev_vel2);
    cudaFree(dev_pos);
    cudaFree(dev_posSorted);

    // TODO-2.1 TODO-2.3 - Free any additional buffers here.
    cudaFree(dev_particleArrayIndices);
    cudaFree(dev_particleGridIndices);
    cudaFree(dev_gridCellStartIndices);
    cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
