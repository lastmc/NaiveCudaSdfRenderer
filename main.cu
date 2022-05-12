#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cuda_device_math.h"

static constexpr auto width = 1280u;
static constexpr auto height = 720u;

static constexpr auto max_ray_depth = 6;
static constexpr auto eps = 1e-4f;
static constexpr auto inf = 1e10f;
static constexpr auto fov = 0.23f;
static constexpr auto dist_limit = 100.0f;
static constexpr auto camera_pos = lc_make_float3(0.0f, 0.32f, 3.7f);
static constexpr auto light_pos = lc_make_float3(-1.5f, 0.6f, 0.3f);
static constexpr auto light_normal = lc_make_float3(1.0f, 0.0f, 0.0f);
static constexpr auto light_radius = 2.0f;

__device__ lc_float intersect_light(lc_float3 pos, lc_float3 d) {
    auto cos_w = lc_dot(-d, light_normal);
    auto dist = lc_dot(d, light_pos - pos);
    auto D = dist / cos_w;
    auto dist_to_center = lc_distance_squared(light_pos, pos + D * d);
    auto valid = cos_w > 0.0f & dist > 0.0f & dist_to_center < light_radius * light_radius;
    return lc_select(inf, D, valid);
}

__device__ lc_uint tea(lc_uint v0, lc_uint v1) {
    lc_uint s0 = 0u;
    for (auto n = 0u; n < 4u; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }
    return v0;
}

__device__ lc_float rand(lc_uint &state) {
    constexpr auto lcg_a = 1664525u;
    constexpr auto lcg_c = 1013904223u;
    state = lcg_a * state + lcg_c;
    // TODO: Probably cast problem
    return static_cast<lc_float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
}

__device__ lc_float3 out_dir(lc_float3 n, lc_uint &seed) {
    auto u = lc_select(
        lc_make_float3(1.f, 0.f, 0.f),
        lc_normalize(lc_cross(n, lc_make_float3(0.0f, 1.0f, 0.0f))),
        abs(n.y) < 1.0f - eps);
    auto v = lc_cross(n, u);
    auto phi = 2.0f * 3.1415926f * rand(seed);
    auto ay = sqrt(rand(seed));
    auto ax = sqrt(1.0f - ay * ay);
    return ax * (cos(phi) * u + sin(phi) * v) + ay * n;
};

__device__ lc_float make_nested(lc_float f) {
    static constexpr auto freq = 40.0f;
    f *= freq;
    f = lc_select(f, lc_select(lc_fract(f), 1.f - lc_fract(f), static_cast<int>(f) % 2 == 0), f < 0.f);
    return (f - 0.2f) * (1.0f / freq);
}

__device__ lc_float sdf(lc_float3 o) {
    auto wall = lc_min(o.y + 0.1f, o.z + 0.4f);
    auto sphere = lc_distance(o, lc_make_float3(0.0f, 0.35f, 0.0f)) - 0.36f;
    auto q = lc_abs(o - lc_make_float3(0.8f, 0.3f, 0.0f)) - 0.3f;
    auto box = lc_length(lc_max(q, lc_make_float3(0.0f))) + lc_min(lc_max(lc_max(q.x, q.y), q.z), 0.0f);
    auto O = o - lc_make_float3(-0.8f, 0.3f, 0.0f);
    auto d = lc_make_float2(lc_length(lc_make_float2(O.x, O.z)) - 0.3f, lc_abs(O.y) - 0.3f);
    auto cylinder = lc_min(lc_max(d.x, d.y), 0.0f) + lc_length(lc_max(d, lc_make_float2(0.0f)));
    auto geometry = make_nested(lc_min(lc_min(sphere, box), cylinder));
    auto g = lc_max(geometry, -(0.32f - (o.y * 0.6f + o.z * 0.8f)));
    return lc_min(wall, g);
};

__device__ lc_float ray_march(lc_float3 p, lc_float3 d) {
    auto dist = 0.0f;
    for(auto j = 0; j < 100; j++) {
        auto s = sdf(p + dist * d);
        if(s <= 1e-6f || dist >= inf) { break; };
        dist += s;
    };
    return lc_min(dist, inf);
}

__device__ lc_float3 sdf_normal(lc_float3 p) {
    static constexpr auto d = 1e-3f;
    auto n = lc_make_float3();
    auto sdf_center = sdf(p);
    for (auto i = 0; i < 3; i++) {
        auto inc = p;
        inc[i] += d;
        n[i] = (1.0f / d) * (sdf(inc) - sdf_center);
    }
    return lc_normalize(n);
}

__device__ void next_hit(lc_float &closest, lc_float3 &normal, lc_float3 &c, lc_float3 pos, lc_float3 d) {
    closest = inf;
    normal = lc_make_float3();
    c = lc_make_float3();
    auto ray_march_dist = ray_march(pos, d);
    if(ray_march_dist < lc_min(dist_limit, closest)) {
        closest = ray_march_dist;
        auto hit_pos = pos + d * closest;
        normal = sdf_normal(hit_pos);
        auto t = static_cast<int>((hit_pos.x + 10.0f) * 1.1f + 0.5f) % 3;
        c = lc_make_float3(0.4f) + lc_make_float3(0.3f, 0.2f, 0.3f) * lc_select(lc_make_float3(0.0f), lc_make_float3(1.0f), t == lc_make_int3(0, 1, 2));
    }
}

__global__ void render_kernel(lc_uint* seed_image, lc_float4* accum_image, lc_uint frame_index) {
    // set_block_size(16u, 8u, 1u);

    auto resolution = lc_make_float2(gridDim.x * blockDim.x, gridDim.y * blockDim.y);
    auto coord = lc_make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    auto global_id = coord.x + coord.y * gridDim.x * blockDim.x;

    if(frame_index == 0u) {
        seed_image[global_id] = tea(coord.x, coord.y);
        accum_image[global_id] = lc_make_float4(lc_make_float3(0.0f), 1.0f);
    }

    auto aspect_ratio = resolution.x / resolution.y;
    auto pos = camera_pos;
    auto seed = seed_image[global_id];
    auto ux = rand(seed);
    auto uy = rand(seed);
    auto uv = lc_make_float2(coord.x + ux, resolution.y - 1u - coord.y + uy);
    auto d = lc_make_float3(
        2.0f * fov * uv / resolution.y - fov * lc_make_float2(aspect_ratio, 1.0f) - 1e-5f, -1.0f);
    d = lc_normalize(d);
    auto throughput = lc_make_float3(1.0f, 1.0f, 1.0f);
    auto hit_light = 0.0f;
    for(auto depth = 0; depth < max_ray_depth; depth++) {
        auto closest = 0.0f;
        auto normal = lc_make_float3();
        auto c = lc_make_float3();
        next_hit(closest, normal, c, pos, d);
        // accum_image[global_id] = lc_make_float4(uv/500.0, 1.0 , 1.0);
        // return;
        auto dist_to_light = intersect_light(pos, d);
        if(dist_to_light < closest) {
            hit_light = 1.0f;
            break;
        }
        if(lc_length_squared(normal) == 0.0f) { break; };
        auto hit_pos = pos + closest * d;
        d = out_dir(normal, seed);
        pos = hit_pos + 1e-4f * d;
        throughput *= c;
    }
    auto accum = accum_image[global_id];
    auto accum_color = lc_make_float3(accum.x, accum.y, accum.z)
        + lc_make_float3(throughput.x, throughput.y, throughput.z) * hit_light;
    accum_image[global_id] = lc_make_float4(accum_color, 1.0f);
    seed_image[global_id] = seed;
}

int main() {

    std::vector<float> pixels(width * height * 4u);
    std::fill(pixels.begin(), pixels.end(), 1.f);

    lc_uint* seedImage;
    lc_float4* accumImage;
    cudaMalloc((void**)&seedImage, sizeof(lc_uint) * width * height);
    cudaMalloc((void**)&accumImage, sizeof(lc_float4) * width * height);

    auto tick = std::chrono::high_resolution_clock::now();

    static constexpr auto totalSpp = 2048;
    dim3 block = make_uint3(16, 8, 1);
    dim3 grid = make_uint3(width/16, height/8, 1);
    for(auto spp = 0; spp < totalSpp; spp++){
        render_kernel<<<grid, block>>>(seedImage, accumImage, spp);
    }

    cudaMemcpy(pixels.data(), accumImage, sizeof(lc_float4) * width * height, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    auto toc = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    printf("Speed = %.2f spp/s\n", totalSpp / ((toc - tick) / 1ns * 1e-6) * 1000);

    for(auto& k: pixels) k /= totalSpp + 1;

    float mean = 0.f;
    for(auto& k: pixels) mean += k;
    mean /= width * height * 4;

    for(auto& k: pixels) k = (k / mean * 0.24);

    stbi_write_hdr("render.hdr", width, height, 4, pixels.data());

}
