// Throwaway research harness: GPU latency for Hyrax-style matrix MSM (BLS12-381).
//
// We have 2^T total scalars laid out as a square-ish matrix:
//   cols = 2^ceil(T/2)  (= msm_size, the SHARED basis vector length)
//   rows = 2^floor(T/2) (= batch_size, number of row-MSMs)
// Odd T => one extra column bit, i.e. more cols than rows.
// Every row is an MSM over the same bases -> batched MSM with shared bases.
//
// Run (CUDA backend must be installed/loadable):
//   cd wrappers/rust/icicle-curves/icicle-bls12-381
//   cargo bench --features cuda_backend --bench hyrax_msm
// Env knobs:
//   PF=1,4,8                 precompute factors to sweep (default 1,4,8)
//   T_LOG2=10                total-scalar-count log2 values (default 10)

use icicle_bls12_381::curve::G1Projective;
use icicle_core::msm::tests::generate_random_affine_points_with_zeroes;
use icicle_core::msm::{msm, precompute_bases, MSMConfig, CUDA_MSM_LARGE_BUCKET_FACTOR};
use icicle_core::projective::Projective;
use icicle_core::traits::GenerateRandom;
use icicle_runtime::{
    device::Device,
    is_device_available,
    memory::{DeviceVec, HostOrDeviceSlice, IntoIcicleSlice, IntoIcicleSliceMut},
    runtime::{load_backend_from_env_or_default, warmup},
    set_device,
    stream::IcicleStream,
};
use std::env;
use std::time::Instant;

type P = G1Projective;
type Scalar = <P as Projective>::ScalarField;
type Affine = <P as Projective>::Affine;

fn env_list_u32(key: &str, default: &[u32]) -> Vec<u32> {
    match env::var(key) {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect(),
        Err(_) => default.to_vec(),
    }
}

fn main() {
    let _ = load_backend_from_env_or_default();
    let target = env::var("BENCH_TARGET").unwrap_or_else(|_| {
        if is_device_available(&Device::new("CUDA", 0)) {
            "CUDA".to_string()
        } else {
            "CPU".to_string()
        }
    });
    let device = Device::new(&target, 0);
    set_device(&device).unwrap();
    println!("# device = {:?}", device);

    let pfs = env_list_u32("PF", &[1, 4, 8]);
    let t_log2s = env_list_u32("T_LOG2", &[10]);

    let mut stream = IcicleStream::create().unwrap();
    warmup(&stream).unwrap();

    // pts_ms   = point-prep: host->device transfer + precompute-table build (shared basis)
    // msm_ms   = compute only, bases already resident on device (msm runs once)
    // total_ms = pts_ms + msm_ms
    // NOTE: scalar host->device transfer is still counted inside msm_ms (scalars
    //       are passed as a host slice). The pts term is purely the basis/point cost.
    println!(
        "{:>4} {:>9} {:>7} {:>4} {:>10} {:>10} {:>10} {:>13}",
        "T", "cols", "rows", "pf", "msm_ms", "pts_ms", "total_ms", "Mscalar/s"
    );

    for &t in &t_log2s {
        let cols = 1usize << ((t + 1) / 2); // 2^ceil(T/2) = msm_size = shared basis length
        let rows = 1usize << (t / 2); // 2^floor(T/2) = batch_size
        let full = rows * cols; // = 2^T

        // shared basis vector of length `cols` without zero points
        let bases = generate_random_affine_points_with_zeroes::<Affine>(cols, 0);

        for &pf in &pfs {
            let pf = pf as usize;

            let mut cfg = MSMConfig::default();
            cfg.stream_handle = *stream;
            cfg.is_async = true;
            cfg.precompute_factor = pf as i32;
            cfg.ext.set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, 5);

            // buffers malloc'd OUTSIDE the timers (assume preallocated/reused).
            // shared basis is precomputed ONCE per (cols, pf) — amortized across all rows.
            let mut precomp = DeviceVec::<Affine>::malloc(pf * cols);

            // scalar matrix, row-major concatenated: [rows x cols] = 2^T scalars
            let scalars = Scalar::generate_random(full);
            let scalars_h = scalars.into_slice();
            let mut results = DeviceVec::<P>::malloc(rows);

            // (A) point-prep: host->device transfer + precompute-table build (shared basis)
            let t_pts = Instant::now();
            precompute_bases::<P>(bases.into_slice(), &cfg, &mut precomp).unwrap();
            stream.synchronize().unwrap();
            let pts_s = t_pts.elapsed().as_secs_f64();

            // (B) compute-only: bases now resident on device (batched msm runs once)
            let t_msm = Instant::now();
            msm(scalars_h, precomp.into_slice(), &cfg, results.into_slice_mut()).unwrap();
            stream.synchronize().unwrap();
            let msm_s = t_msm.elapsed().as_secs_f64();

            let total_s = pts_s + msm_s;
            let msm_ms = msm_s * 1e3;
            let pts_ms = pts_s * 1e3;
            let total_ms = total_s * 1e3;
            let mscalar_s = (full as f64) / msm_s / 1e6; // compute-only throughput
            println!(
                "{:>4} {:>9} {:>7} {:>4} {:>10.3} {:>10.3} {:>10.3} {:>13.1}",
                t, cols, rows, pf, msm_ms, pts_ms, total_ms, mscalar_s
            );
        }
    }

    stream.destroy().unwrap();
}
