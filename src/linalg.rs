//! Linear algebra primitives: Gaussian elimination, Jacobi eigenvalue method.
//! No external math dependencies — everything from scratch.

/// Solve Ax = b via Gaussian elimination with partial pivoting.
pub fn gaussian_eliminate(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }
    for row in a {
        if row.len() != n {
            return None;
        }
    }

    // Augmented matrix
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = a[i].clone();
        row.push(b[i]);
        aug.push(row);
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for r in (col + 1)..n {
            if aug[r][col].abs() > max_val {
                max_val = aug[r][col].abs();
                max_row = r;
            }
        }
        if max_val < 1e-12 {
            return None; // Singular
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for r in (col + 1)..n {
            let factor = aug[r][col] / pivot;
            for c in col..=n {
                aug[r][c] -= factor * aug[col][c];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-12 {
            return None;
        }
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    Some(x)
}

/// Compute eigenvalues of a real symmetric matrix using the Jacobi eigenvalue method.
/// Returns eigenvalues sorted in ascending order.
pub fn jacobi_eigenvalues(mat: &[Vec<f64>], max_iters: usize) -> Vec<f64> {
    let n = mat.len();
    if n == 0 {
        return vec![];
    }
    // Work on a copy
    let mut a: Vec<Vec<f64>> = mat.to_vec();

    // Repeatedly apply Givens rotations to zero off-diagonal elements
    for _ in 0..max_iters {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-10 {
            break; // Converged
        }

        // Compute rotation angle
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation: A' = G^T A G
        // Update rows/cols p and q
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                new_a[i][p] = c * aip + s * aiq;
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = -s * aip + c * aiq;
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_a[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;

        a = new_a;
    }

    // Extract diagonal as eigenvalues
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Matrix multiplication for square matrices.
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

/// Transpose a square matrix.
pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut t = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            t[i][j] = a[j][i];
        }
    }
    t
}

/// Compute the trace of a square matrix.
pub fn trace(a: &[Vec<f64>]) -> f64 {
    (0..a.len()).map(|i| a[i][i]).sum()
}

/// Compute the Frobenius norm of a matrix.
pub fn frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 5.0];
        let x = gaussian_eliminate(&a, &b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-9);
        assert!((x[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_2x2() {
        let a = vec![vec![2.0, 1.0], vec![5.0, 3.0]];
        let b = vec![11.0, 27.0]; // x=6, y=-1 => 12-1=11, 30-3=27 ✓ (no: 2*6+(-1)=11, 5*6+3*(-1)=27) => no. x=6,y=-1 => 2*6+(-1)=11, 30-3=27 => y=-1, so b=27, 5*6+3*(-1)=27 ✓
        // Actually let me recheck: 2x+y=11, 5x+3y=27 => from first y=11-2x => 5x+33-6x=27 => -x=-6 => x=6, y=-1
        let x = gaussian_eliminate(&a, &b).unwrap();
        assert!((x[0] - 6.0).abs() < 1e-9);
        assert!((x[1] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![3.0, 6.0];
        assert!(gaussian_eliminate(&a, &b).is_none());
    }

    #[test]
    fn test_jacobi_identity() {
        let a = vec![vec![3.0, 0.0], vec![0.0, 7.0]];
        let eigs = jacobi_eigenvalues(&a, 100);
        assert!((eigs[0] - 3.0).abs() < 1e-6);
        assert!((eigs[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_jacobi_2x2() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let eigs = jacobi_eigenvalues(&a, 100);
        // Eigenvalues: (7 ± sqrt(1+4))/2 = (7 ± sqrt(5))/2... no.
        // λ = (7 ± sqrt(7²-4*11))/2... no. det(A-λI) = (4-λ)(3-λ)-1 = λ²-7λ+11 = 0
        // λ = (7 ± sqrt(49-44))/2 = (7 ± sqrt(5))/2
        let l1 = (7.0 - 5.0_f64.sqrt()) / 2.0;
        let l2 = (7.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((eigs[0] - l1).abs() < 1e-6);
        assert!((eigs[1] - l2).abs() < 1e-6);
    }

    #[test]
    fn test_jacobi_3x3() {
        let a = vec![vec![2.0, -1.0, 0.0], vec![-1.0, 2.0, -1.0], vec![0.0, -1.0, 2.0]];
        let eigs = jacobi_eigenvalues(&a, 200);
        // Path graph P3 eigenvalues: 2-√2, 2, 2+√2
        let sqrt2 = 2.0_f64.sqrt();
        assert!((eigs[0] - (2.0 - sqrt2)).abs() < 1e-4);
        assert!((eigs[1] - 2.0).abs() < 1e-4);
        assert!((eigs[2] - (2.0 + sqrt2)).abs() < 1e-4);
    }

    #[test]
    fn test_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((trace(&a) - 5.0).abs() < 1e-9);
    }
}
