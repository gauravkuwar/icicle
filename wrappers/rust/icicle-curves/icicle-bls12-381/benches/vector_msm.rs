// Throwaway research harness: GPU latency for a single (vector) MSM (BLS12-381).
//
// A "vector" MSM is the non-batched case: one length-N basis vector and one
// length-N scalar vector produce ONE output point:
//   result = sum_i scalars[i] * bases[i]
// This is batch_size = 1, the degenerate case of the batched (matrix) MSM in
// hyrax_msm.rs. We sweep the precompute factor the same way to see how
// precompute amortizes for a single MSM of a given length.
//
// Run (CUDA backend must be installed/loadable):
//   cd wrappers/rust/icicle-curves/icicle-bls12-381
//   cargo bench --features cuda_backend --bench vector_msm
// Env knobs:
//   PF=1,4,8                 precompute factors to sweep (default 1,4,8)
//   N_LOG2=18,20,22          MSM-length log2 values to sweep (default 18,20,22)

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
    let n_log2s = env_list_u32("N_LOG2", &[18, 20, 22]);

    let mut stream = IcicleStream::create().unwrap();
    warmup(&stream).unwrap();

    println!(
        "{:>4} {:>10} {:>4} {:>10} {:>13}",
        "N", "size", "pf", "lat_ms", "Mscalar/s"
    );

    for &n in &n_log2s {
        let size = 1usize << n; // MSM length = basis length = scalar count

        // basis vector of length `size` without zero points
        let bases = generate_random_affine_points_with_zeroes::<Affine>(size, 0);

        for &pf in &pfs {
            let pf = pf as usize;

            let mut cfg = MSMConfig::default();
            cfg.stream_handle = *stream;
            cfg.is_async = true;
            cfg.precompute_factor = pf as i32;
            cfg.ext.set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, 5);

            // precompute the basis ONCE per (size, pf)
            let mut precomp = DeviceVec::<Affine>::malloc(pf * size);
            precompute_bases::<P>(bases.into_slice(), &cfg, &mut precomp).unwrap();

            // single scalar vector of length `size`
            let scalars = Scalar::generate_random(size);
            let scalars_h = scalars.into_slice();

            // batch_size = 1 => a single output point
            let mut result = DeviceVec::<P>::malloc(1);

            // single timed MSM call (sync to capture full GPU time)
            let timer = Instant::now();
            msm(scalars_h, precomp.into_slice(), &cfg, result.into_slice_mut()).unwrap();
            stream.synchronize().unwrap();
            let per_call = timer.elapsed().as_secs_f64();

            let lat_ms = per_call * 1e3;
            let mscalar_s = (size as f64) / per_call / 1e6;
            println!(
                "{:>4} {:>10} {:>4} {:>10.3} {:>13.1}",
                n, size, pf, lat_ms, mscalar_s
            );
        }
    }

    stream.destroy().unwrap();
}
