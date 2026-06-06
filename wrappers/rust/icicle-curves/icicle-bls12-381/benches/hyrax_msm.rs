// Research harness: GPU latency for Hyrax-style batched MSM (BLS12-381).
// 2^T scalars as a matrix: cols = 2^ceil(T/2) (shared basis len), rows = 2^floor(T/2)
// (batch size). Every row is an MSM over the same bases -> shared-basis batched MSM.
// Sweeps window bits `c` and precompute factor `pf` to find the optimal config.
//
//   cargo bench --features cuda_backend --bench hyrax_msm

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

const T_LOG2S: &[u32] = &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25];
const WINDOW_SIZES: &[i32] = &[4, 5, 6, 7, 8, 9, 10, 11, 12];
const PRECOMPUTE_FACTORS: &[i32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

fn main() {
    let _ = load_backend_from_env_or_default();
    let target = env::var("BENCH_TARGET").unwrap_or_else(|_| {
        if is_device_available(&Device::new("CUDA", 0)) {
            "CUDA".to_string()
        } else {
            "CPU".to_string()
        }
    });
    set_device(&Device::new(&target, 0)).unwrap();
    println!("# device = {}", target);

    let mut stream = IcicleStream::create().unwrap();
    warmup(&stream).unwrap();

    // pts_ms = host->device transfer + precompute-table build; msm_ms = compute only
    // (bases resident); total_ms = pts_ms + msm_ms. Scalar transfer is inside msm_ms.
    println!(
        "{:>4} {:>9} {:>7} {:>3} {:>4} {:>10} {:>10} {:>10} {:>13}",
        "T", "cols", "rows", "c", "pf", "msm_ms", "pts_ms", "total_ms", "Mscalar/s"
    );

    for &t in T_LOG2S {
        let log_cols = (t + 1) / 2; // log2(cols) = ceil(T/2)
        let cols = 1usize << log_cols;
        let rows = 1usize << (t / 2);
        let full = rows * cols;
        let bases = generate_random_affine_points_with_zeroes::<Affine>(cols, 0);

        for &c in WINDOW_SIZES {
            for &pf in PRECOMPUTE_FACTORS {
                let mut cfg = MSMConfig::default();
                cfg.stream_handle = *stream;
                cfg.is_async = true;
                cfg.c = c;
                cfg.precompute_factor = pf;
                cfg.ext.set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, 10);

                let scalars = Scalar::generate_random(full);
                let scalars_h = scalars.into_slice();
                let mut precomp = DeviceVec::<Affine>::malloc(pf as usize * cols);
                let mut results = DeviceVec::<P>::malloc(rows);

                // (A) point-prep
                let t_pts = Instant::now();
                precompute_bases::<P>(bases.into_slice(), &cfg, &mut precomp).unwrap();
                stream.synchronize().unwrap();
                let pts_s = t_pts.elapsed().as_secs_f64();

                // (B) compute only (bases resident; derives batch_size from results len)
                let t_msm = Instant::now();
                msm(scalars_h, precomp.into_slice(), &cfg, results.into_slice_mut()).unwrap();
                stream.synchronize().unwrap();
                let msm_s = t_msm.elapsed().as_secs_f64();

                println!(
                    "{:>4} {:>9} {:>7} {:>3} {:>4} {:>10.3} {:>10.3} {:>10.3} {:>13.1}",
                    t, cols, rows, c, pf,
                    msm_s * 1e3, pts_s * 1e3, (pts_s + msm_s) * 1e3,
                    full as f64 / msm_s / 1e6,
                );
            }
        }
    }

    stream.destroy().unwrap();
}
