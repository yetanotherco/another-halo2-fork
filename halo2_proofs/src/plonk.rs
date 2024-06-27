//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use blake2b_simd::Params as Blake2bParams;
use group::ff::{Field, FromUniformBytes, PrimeField};
use serde::Serialize;

use crate::arithmetic::CurveAffine;
use crate::helpers::{
    polynomial_slice_byte_length, read_polynomial_vec, write_polynomial_slice, SerdeCurveAffine,
    SerdePrimeField,
};
use crate::poly::{Coeff, EvaluationDomain, LagrangeCoeff, PinnedEvaluationDomain, Polynomial};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript};
use crate::SerdeFormat;

mod assigned;
mod circuit;
mod error;
mod evaluation;
mod keygen;
mod lookup;
pub mod permutation;
// mod shuffle;
mod vanishing;

mod prover;
mod verifier;

pub use assigned::*;
pub use circuit::*;
pub use error::*;
pub use keygen::*;
pub use prover::*;
pub use verifier::*;

use evaluation::Evaluator;

use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Writes ConstraintSystemBack, VerifyingKey, and ProverParams to a file to be sent to aligned.
pub fn write_params<C: CurveAffine>(
    params_buf: &[u8],
    cs_buf: &[u8],
    vk_buf: &[u8],
    params_path: &str,
) -> Result<(), Error>
where
    <C as CurveAffine>::ScalarExt: Serialize,
{
    let vk_len = vk_buf.len();
    let params_len = params_buf.len();

    //Write everything to parameters file
    let params_file = File::create(params_path).unwrap();
    let mut writer = BufWriter::new(params_file);
    //Write Parameter Lengths as u32
    writer
        .write_all(&(cs_buf.len() as u32).to_le_bytes())
        .unwrap();
    writer.write_all(&(vk_len as u32).to_le_bytes()).unwrap();
    writer
        .write_all(&(params_len as u32).to_le_bytes())
        .unwrap();
    //Write Parameters
    writer.write_all(&cs_buf).unwrap();
    writer.write_all(&vk_buf).unwrap();
    writer.write_all(&params_buf).unwrap();
    writer.flush().unwrap();
    Ok(())
}

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<C: CurveAffine> {
    domain: EvaluationDomain<C::Scalar>,
    fixed_commitments: Vec<C>,
    permutation: permutation::VerifyingKey<C>,
    cs: ConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: C::Scalar,
    selectors: Vec<Vec<bool>>,
    /// Whether selector compression is turned on or not.
    compress_selectors: bool,
}

impl<C: SerdeCurveAffine> VerifyingKey<C>
where
    C::Scalar: SerdePrimeField + FromUniformBytes<64>, // the FromUniformBytes<64> should not be necessary: currently serialization always stores a Blake2b hash of verifying key; this should be removed
{
    /// Writes a verifying key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in standard form.
    /// Writes a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    /// Writes a field element into raw bytes in its internal Montgomery representation,
    /// WITHOUT performing the expensive Montgomery reduction.
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        // Version byte that will be checked on read.
        writer.write_all(&[0x02])?;
        writer.write_all(&self.domain.k().to_le_bytes())?;
        writer.write_all(&[self.compress_selectors as u8])?;
        writer.write_all(&(self.fixed_commitments.len() as u32).to_le_bytes())?;
        for commitment in &self.fixed_commitments {
            commitment.write(writer, format)?;
        }
        self.permutation.write(writer, format)?;

        if !self.compress_selectors {
            assert!(self.selectors.is_empty());
        }
        // write self.selectors
        for selector in &self.selectors {
            // since `selector` is filled with `bool`, we pack them 8 at a time into bytes and then write
            for bits in selector.chunks(8) {
                writer.write_all(&[crate::helpers::pack(bits)])?;
            }
        }
        Ok(())
    }

    pub fn read_cs<R: io::Read>(
        reader: &mut R,
        format: SerdeFormat,
        cs: ConstraintSystem<C::Scalar>,
    ) -> io::Result<Self> {
        let mut version_byte = [0u8; 1];
        reader.read_exact(&mut version_byte)?;
        if 0x02 != version_byte[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected version byte",
            ));
        }
        let mut k = [0u8; 4];
        reader.read_exact(&mut k)?;
        let k = u32::from_le_bytes(k);
        let mut compress_selectors = [0u8; 1];
        reader.read_exact(&mut compress_selectors)?;
        if compress_selectors[0] != 0 && compress_selectors[0] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected compress_selectors not boolean",
            ));
        }
        let compress_selectors = compress_selectors[0] == 1;
        let degree = cs.degree();
        let domain = EvaluationDomain::new(degree as u32, k);

        let mut num_fixed_columns = [0u8; 4];
        reader.read_exact(&mut num_fixed_columns)?;
        let num_fixed_columns = u32::from_le_bytes(num_fixed_columns);

        let fixed_commitments: Vec<_> = (0..num_fixed_columns)
            .map(|_| C::read(reader, format))
            .collect::<io::Result<_>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation, format)?;

        let (cs, selectors) = if compress_selectors {
            // read selectors
            let selectors: Vec<Vec<bool>> = vec![vec![false; 1 << k]; cs.num_selectors]
                .into_iter()
                .map(|mut selector| {
                    let mut selector_bytes = vec![0u8; (selector.len() + 7) / 8];
                    reader.read_exact(&mut selector_bytes)?;
                    for (bits, byte) in selector.chunks_mut(8).zip(selector_bytes) {
                        crate::helpers::unpack(byte, bits);
                    }
                    Ok(selector)
                })
                .collect::<io::Result<_>>()?;
            let (cs, _) = cs.compress_selectors(selectors.clone());
            (cs, selectors)
        } else {
            // we still need to replace selectors with fixed Expressions in `cs`
            let fake_selectors = vec![vec![false]; cs.num_selectors];
            let (cs, _) = cs.directly_convert_selectors_to_fixed(fake_selectors);
            (cs, vec![])
        };

        Ok(Self::from_parts(
            domain,
            fixed_commitments,
            permutation,
            cs,
            selectors,
            compress_selectors,
        ))
    }

    /// Reads a verification key from a buffer.
    ///
    /// Reads a curve element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    /// Reads a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    /// Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    /// does not perform any checks
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let mut version_byte = [0u8; 1];
        reader.read_exact(&mut version_byte)?;
        if 0x02 != version_byte[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected version byte",
            ));
        }
        let mut k = [0u8; 4];
        reader.read_exact(&mut k)?;
        let k = u32::from_le_bytes(k);
        let mut compress_selectors = [0u8; 1];
        reader.read_exact(&mut compress_selectors)?;
        if compress_selectors[0] != 0 && compress_selectors[0] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected compress_selectors not boolean",
            ));
        }
        let compress_selectors = compress_selectors[0] == 1;
        let (domain, cs, _) = keygen::create_domain::<C, ConcreteCircuit>(
            k,
            #[cfg(feature = "circuit-params")]
            params,
        );
        let mut num_fixed_columns = [0u8; 4];
        reader.read_exact(&mut num_fixed_columns)?;
        let num_fixed_columns = u32::from_le_bytes(num_fixed_columns);

        let fixed_commitments: Vec<_> = (0..num_fixed_columns)
            .map(|_| C::read(reader, format))
            .collect::<io::Result<_>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation, format)?;

        let (cs, selectors) = if compress_selectors {
            // read selectors
            let selectors: Vec<Vec<bool>> = vec![vec![false; 1 << k]; cs.num_selectors]
                .into_iter()
                .map(|mut selector| {
                    let mut selector_bytes = vec![0u8; (selector.len() + 7) / 8];
                    reader.read_exact(&mut selector_bytes)?;
                    for (bits, byte) in selector.chunks_mut(8).zip(selector_bytes) {
                        crate::helpers::unpack(byte, bits);
                    }
                    Ok(selector)
                })
                .collect::<io::Result<_>>()?;
            let (cs, _) = cs.compress_selectors(selectors.clone());
            (cs, selectors)
        } else {
            // we still need to replace selectors with fixed Expressions in `cs`
            let fake_selectors = vec![vec![false]; cs.num_selectors];
            let (cs, _) = cs.directly_convert_selectors_to_fixed(fake_selectors);
            (cs, vec![])
        };

        Ok(Self::from_parts(
            domain,
            fixed_commitments,
            permutation,
            cs,
            selectors,
            compress_selectors,
        ))
    }

    /// Writes a verifying key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length());
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a verification key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<C::Scalar>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    fn bytes_length(&self) -> usize {
        8 + (self.fixed_commitments.len() * C::default().to_bytes().as_ref().len())
            + self.permutation.bytes_length()
            + self.selectors.len()
                * (self
                    .selectors
                    .get(0)
                    .map(|selector| (selector.len() + 7) / 8)
                    .unwrap_or(0))
    }

    fn from_parts(
        domain: EvaluationDomain<C::Scalar>,
        fixed_commitments: Vec<C>,
        permutation: permutation::VerifyingKey<C>,
        cs: ConstraintSystem<C::Scalar>,
        selectors: Vec<Vec<bool>>,
        compress_selectors: bool,
    ) -> Self
    where
        C::Scalar: FromUniformBytes<64>,
    {
        // Compute cached values.
        let cs_degree = cs.degree();

        let mut vk = Self {
            domain,
            fixed_commitments,
            permutation,
            cs,
            cs_degree,
            // Temporary, this is not pinned.
            transcript_repr: C::Scalar::ZERO,
            selectors,
            compress_selectors,
        };

        let mut hasher = Blake2bParams::new()
            .hash_length(64)
            .personal(b"Halo2-Verify-Key")
            .to_state();

        let s = format!("{:?}", vk.pinned());

        hasher.update(&(s.len() as u64).to_le_bytes());
        hasher.update(s.as_bytes());

        // Hash in final Blake2bState
        vk.transcript_repr = C::Scalar::from_uniform_bytes(hasher.finalize().as_array());

        vk
    }

    /// Hashes a verification key into a transcript.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, C> {
        PinnedVerificationKey {
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }

    /// Returns commitments of fixed polynomials
    pub fn fixed_commitments(&self) -> &Vec<C> {
        &self.fixed_commitments
    }

    /// Returns `VerifyingKey` of permutation
    pub fn permutation(&self) -> &permutation::VerifyingKey<C> {
        &self.permutation
    }

    /// Returns `ConstraintSystem`
    pub fn cs(&self) -> &ConstraintSystem<C::Scalar> {
        &self.cs
    }

    /// Returns representative of this `VerifyingKey` in transcripts
    pub fn transcript_repr(&self) -> C::Scalar {
        self.transcript_repr
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, C: CurveAffine> {
    base_modulus: &'static str,
    scalar_modulus: &'static str,
    domain: PinnedEvaluationDomain<'a, C::Scalar>,
    cs: PinnedConstraintSystem<'a, C::Scalar>,
    fixed_commitments: &'a Vec<C>,
    permutation: &'a permutation::VerifyingKey<C>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<C: CurveAffine> {
    vk: VerifyingKey<C>,
    l0: Polynomial<C::Scalar, Coeff>,
    l_last: Polynomial<C::Scalar, Coeff>,
    l_active_row: Polynomial<C::Scalar, Coeff>,
    fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    permutation: permutation::ProvingKey<C>,
    ev: Evaluator<C>,
}

impl<C: CurveAffine> ProvingKey<C>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<C> {
        &self.vk
    }

    /// Gets the total number of bytes in the serialization of `self`
    fn bytes_length(&self) -> usize {
        let scalar_len = C::Scalar::default().to_repr().as_ref().len();
        self.vk.bytes_length()
            + 12
            + scalar_len * (self.l0.len() + self.l_last.len() + self.l_active_row.len())
            + polynomial_slice_byte_length(&self.fixed_values)
            + polynomial_slice_byte_length(&self.fixed_polys)
            + self.permutation.bytes_length()
    }
}

impl<C: SerdeCurveAffine> ProvingKey<C>
where
    C::Scalar: SerdePrimeField + FromUniformBytes<64>,
{
    /// Writes a proving key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in standard form.
    /// Writes a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    /// Writes a field element into raw bytes in its internal Montgomery representation,
    /// WITHOUT performing the expensive Montgomery reduction.
    /// Does so by first writing the verifying key and then serializing the rest of the data (in the form of field polynomials)
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        self.vk.write(writer, format)?;
        self.l0.write(writer, format);
        self.l_last.write(writer, format);
        self.l_active_row.write(writer, format);
        write_polynomial_slice(&self.fixed_values, writer, format);
        write_polynomial_slice(&self.fixed_polys, writer, format);
        self.permutation.write(writer, format);
        Ok(())
    }

    /// Reads a proving key from a buffer.
    /// Does so by reading verification key first, and then deserializing the rest of the file into the remaining proving key data.
    ///
    /// Reads a curve element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    /// Reads a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    /// Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    /// does not perform any checks
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let vk = VerifyingKey::<C>::read::<R, ConcreteCircuit>(
            reader,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )?;
        let l0 = Polynomial::read(reader, format);
        let l_last = Polynomial::read(reader, format);
        let l_active_row = Polynomial::read(reader, format);
        let fixed_values = read_polynomial_vec(reader, format);
        let fixed_polys = read_polynomial_vec(reader, format);
        let permutation = permutation::ProvingKey::read(reader, format);
        let ev = Evaluator::new(vk.cs());
        Ok(Self {
            vk,
            l0,
            l_last,
            l_active_row,
            fixed_values,
            fixed_polys,
            permutation,
            ev,
        })
    }

    /// Writes a proving key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length());
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a proving key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<C::Scalar>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<C::Scalar> {
        &self.domain
    }
}

#[derive(Clone, Copy, Debug)]
struct Theta;
type ChallengeTheta<F> = ChallengeScalar<F, Theta>;

#[derive(Clone, Copy, Debug)]
struct Beta;
type ChallengeBeta<F> = ChallengeScalar<F, Beta>;

#[derive(Clone, Copy, Debug)]
struct Gamma;
type ChallengeGamma<F> = ChallengeScalar<F, Gamma>;

#[derive(Clone, Copy, Debug)]
struct Y;
type ChallengeY<F> = ChallengeScalar<F, Y>;

#[derive(Clone, Copy, Debug)]
struct X;
type ChallengeX<F> = ChallengeScalar<F, X>;

#[cfg(test)]
mod tests {
    use super::io::BufReader;
    use halo2curves::bn256::{Bn256, Fr, G1Affine};

    use crate::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        poly::{
            commitment::Params,
            kzg::{multiopen::VerifierSHPLONK, strategy::SingleStrategy},
            Rotation,
        },
    };

    #[test]
    fn test_proof_serialization() {
        use super::*;

        use crate::{
            plonk::{create_proof, keygen_pk, keygen_vk_custom, verify_proof},
            poly::kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::ProverSHPLONK,
            },
            transcript::{
                Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer,
                TranscriptWriterBuffer,
            },
        };
        use ff::{Field, PrimeField};
        use rand_core::OsRng;
        use std::{
            fs::File,
            io::{BufWriter, ErrorKind, Read, Write},
        };

        fn read_fr(mut buf: &[u8]) -> Result<Vec<Fr>, ErrorKind> {
            let mut instances = Vec::with_capacity(buf.len() / 32);
            // Buffer to store each 32-byte slice
            let mut buffer = [0; 32];

            loop {
                // Read 32 bytes into the buffer
                match buf.read_exact(&mut buffer) {
                    Ok(_) => {
                        instances.push(Fr::from_bytes(&buffer).unwrap());
                    }
                    Err(ref e) if e.kind() == ErrorKind::UnexpectedEof => {
                        // If end of file reached, break the loop
                        break;
                    }
                    Err(e) => {
                        eprintln!("Error Deserializing Public Inputs: {}", e);
                        return Err(ErrorKind::Other);
                    }
                }
            }

            Ok(instances)
        }

        #[derive(Clone, Copy)]
        struct StandardPlonkConfig {
            a: Column<Advice>,
            b: Column<Advice>,
            c: Column<Advice>,
            q_a: Column<Fixed>,
            q_b: Column<Fixed>,
            q_c: Column<Fixed>,
            q_ab: Column<Fixed>,
            constant: Column<Fixed>,
            #[allow(dead_code)]
            instance: Column<Instance>,
        }

        impl StandardPlonkConfig {
            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
                let [a, b, c] = [(); 3].map(|_| meta.advice_column());
                let [q_a, q_b, q_c, q_ab, constant] = [(); 5].map(|_| meta.fixed_column());
                let instance = meta.instance_column();

                [a, b, c].map(|column| meta.enable_equality(column));

                meta.create_gate(
                    "q_a·a + q_b·b + q_c·c + q_ab·a·b + constant + instance = 0",
                    |meta| {
                        let [a, b, c] =
                            [a, b, c].map(|column| meta.query_advice(column, Rotation::cur()));
                        let [q_a, q_b, q_c, q_ab, constant] = [q_a, q_b, q_c, q_ab, constant]
                            .map(|column| meta.query_fixed(column, Rotation::cur()));
                        let instance = meta.query_instance(instance, Rotation::cur());
                        Some(
                            q_a * a.clone()
                                + q_b * b.clone()
                                + q_c * c
                                + q_ab * a * b
                                + constant
                                + instance,
                        )
                    },
                );

                StandardPlonkConfig {
                    a,
                    b,
                    c,
                    q_a,
                    q_b,
                    q_c,
                    q_ab,
                    constant,
                    instance,
                }
            }
        }

        #[derive(Clone, Default)]
        struct StandardPlonk(Fr);

        impl Circuit<Fr> for StandardPlonk {
            type Config = StandardPlonkConfig;
            type FloorPlanner = SimpleFloorPlanner;
            #[cfg(feature = "circuit-params")]
            type Params = ();

            fn without_witnesses(&self) -> Self {
                Self::default()
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                StandardPlonkConfig::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), Error> {
                layouter.assign_region(
                    || "",
                    |mut region| {
                        region.assign_advice(config.a, 0, Value::known(self.0));
                        region.assign_fixed(config.q_a, 0, -Fr::one());

                        region.assign_advice(config.a, 1, Value::known(-Fr::from(5u64)));
                        for (idx, column) in (1..).zip([
                            config.q_a,
                            config.q_b,
                            config.q_c,
                            config.q_ab,
                            config.constant,
                        ]) {
                            region.assign_fixed(column, 1, Fr::from(idx as u64));
                        }

                        let a = region.assign_advice(config.a, 2, Value::known(Fr::one()));
                        a.copy_advice(&mut region, config.b, 3);
                        a.copy_advice(&mut region, config.c, 4);
                        Ok(())
                    },
                )
            }
        }

        let circuit = StandardPlonk(Fr::random(OsRng));
        let params = ParamsKZG::setup(4, OsRng);
        let compress_selectors = true;
        let vk =
            keygen_vk_custom(&params, &circuit, compress_selectors).expect("vk should not fail");
        let cs = vk.cs();
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("pk should not fail");
        let instances: &[&[Fr]] = &[&[circuit.0]];

        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            Challenge255<G1Affine>,
            _,
            Blake2bWrite<Vec<u8>, G1Affine, Challenge255<_>>,
            _,
        >(
            &params,
            &pk,
            &[circuit.clone()],
            &[instances],
            OsRng,
            &mut transcript,
        )
        .expect("prover should not fail");
        let proof = transcript.finalize();

        //write public input
        let f = File::create("public_input.bin").unwrap();
        let mut writer = BufWriter::new(f);
        instances.to_vec().into_iter().flatten().for_each(|fp| {
            writer.write(&fp.to_repr()).unwrap();
        });
        writer.flush().unwrap();

        //write proof
        let f = File::create("proof.bin").unwrap();
        let mut writer = BufWriter::new(f);
        writer.write(&proof).unwrap();
        writer.flush().unwrap();

        let cs_buf = bincode::serialize(cs).unwrap();

        let mut vk_buf = Vec::new();
        vk.write(&mut vk_buf, SerdeFormat::RawBytes).unwrap();

        // write cs, vk, params
        let mut params_buf = Vec::new();
        params.write(&mut params_buf).unwrap();
        write_params::<G1Affine>(&params_buf, &cs_buf, &vk_buf, "params.bin").unwrap();

        // Read Instances
        let mut f = File::open("public_input.bin").unwrap();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        let res = read_fr(&buf).unwrap();
        let instances = res.as_slice();

        // Read Proof
        let mut f = File::open("proof.bin").unwrap();
        let mut proof = Vec::new();
        f.read_to_end(&mut proof).unwrap();

        // Read Params
        let mut f = File::open("params.bin").unwrap();
        let mut params_buf = Vec::new();
        f.read_to_end(&mut params_buf).unwrap();

        // Select Constraint System Bytes
        let cs_len_buf: [u8; 4] = params_buf[..4]
            .try_into()
            .map_err(|_| "Failed to convert slice to [u8; 4]")
            .unwrap();
        let cs_len = u32::from_le_bytes(cs_len_buf) as usize;
        let mut cs_buffer = vec![0u8; cs_len];
        let cs_offset = 12;
        cs_buffer[..cs_len].clone_from_slice(&params_buf[cs_offset..(cs_offset + cs_len)]);

        // Select Verifier Key Bytes
        let vk_len_buf: [u8; 4] = params_buf[4..8]
            .try_into()
            .map_err(|_| "Failed to convert slice to [u8; 4]")
            .unwrap();
        let vk_len = u32::from_le_bytes(vk_len_buf) as usize;
        let mut vk_buffer = vec![0u8; vk_len];
        let vk_offset = cs_offset + cs_len;
        vk_buffer[..vk_len].clone_from_slice(&params_buf[vk_offset..(vk_offset + vk_len)]);
        let mut vk_reader = &mut BufReader::new(vk_buffer.as_slice());

        // Select KZG Params Bytes
        let kzg_len_buf: [u8; 4] = params_buf[8..12]
            .try_into()
            .map_err(|_| "Failed to convert slice to [u8; 4]")
            .unwrap();
        let kzg_params_len = u32::from_le_bytes(kzg_len_buf) as usize;
        let mut kzg_params_buffer = vec![0u8; kzg_params_len];
        let kzg_offset = vk_offset + vk_len;
        kzg_params_buffer[..kzg_params_len].clone_from_slice(&params_buf[kzg_offset..]);

        let cs = bincode::deserialize(&cs_buffer).unwrap();

        let vk =
            VerifyingKey::<G1Affine>::read_cs(&mut vk_reader, SerdeFormat::RawBytes, cs)
                .unwrap();
        let params =
            Params::read::<_>(&mut BufReader::new(&kzg_params_buffer[..kzg_params_len])).unwrap();

        let strategy = SingleStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        assert!(verify_proof::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<'_, Bn256>,
            Challenge255<G1Affine>,
            Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
            SingleStrategy<'_, Bn256>,
        >(&params, &vk, strategy, &[&[instances]], &mut transcript)
        .is_ok());
    }
}
