// ============================================================
// Matrix Operations — Gauss Elimination with Partial Pivoting
// ============================================================

/**
 * Performs Gaussian Elimination with partial pivoting to solve Ax = b.
 * @param A - Coefficient matrix (n x n)
 * @param b - Constants vector (n)
 * @returns Solution vector x, or null if the matrix is singular
 */
export function gaussElimination(A: number[][], b: number[]): number[] | null {
    const n = A.length;
    // Create augmented matrix [A|b]
    const aug: number[][] = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
        // Partial pivoting: find the row with the largest absolute value in this column
        let maxRow = col;
        let maxVal = Math.abs(aug[col][col]);
        for (let row = col + 1; row < n; row++) {
            const val = Math.abs(aug[row][col]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = row;
            }
        }

        // Swap rows
        if (maxRow !== col) {
            [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
        }

        // Check for singular matrix
        if (Math.abs(aug[col][col]) < 1e-12) {
            return null;
        }

        // Forward elimination
        for (let row = col + 1; row < n; row++) {
            const factor = aug[row][col] / aug[col][col];
            for (let j = col; j <= n; j++) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        let sum = aug[i][n];
        for (let j = i + 1; j < n; j++) {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    return x;
}

/**
 * Multiplies a matrix by a vector: result = A * v
 */
export function matVecMultiply(A: number[][], v: number[]): number[] {
    return A.map(row => row.reduce((sum, val, j) => sum + val * v[j], 0));
}

/**
 * Transposes a matrix
 */
export function transpose(A: number[][]): number[][] {
    if (A.length === 0) return [];
    const rows = A.length;
    const cols = A[0].length;
    const result: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

/**
 * Computes the Euclidean distance between two vectors.
 */
export function euclideanDistance(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0));
}
