// ============================================================
// SVM Regressor — Simplified SVR with SMO-like approach
// ============================================================

import { TrainableModel, ModelState, HyperparameterDef } from './base.model';

export class SVMModel implements TrainableModel {
    readonly name = 'SVM';
    readonly category = 'regression';
    readonly description = 'Support Vector Machine (SVR) — Regresión con vectores de soporte';
    readonly hyperparameterSchema: HyperparameterDef[] = [
        { name: 'C', label: 'Parámetro C (Regularización)', type: 'float', min: 0.01, max: 100, step: 0.1, default: 1.0 },
        { name: 'epsilon', label: 'Epsilon (Margen)', type: 'float', min: 0.001, max: 1, step: 0.01, default: 0.1 },
        { name: 'kernel', label: 'Kernel', type: 'select', default: 0, options: [{ value: 0, label: 'Lineal' }, { value: 1, label: 'RBF' }] },
        { name: 'gamma', label: 'Gamma (RBF)', type: 'float', min: 0.001, max: 10, step: 0.01, default: 0.1 }
    ];

    train(X: number[][], y: number[], params: Record<string, number>): ModelState {
        const C = params.C || 1.0;
        const epsilon = params.epsilon || 0.1;
        const kernelType = params.kernel || 0;
        const gamma = params.gamma || 0.1;
        const n = X.length;
        const features = X[0]?.length || 1;

        // For linear kernel, use gradient descent to find weights
        if (kernelType === 0) {
            return this.trainLinear(X, y, C, epsilon, features);
        }

        // For RBF kernel, store training data and use kernel trick
        return this.trainRBF(X, y, C, epsilon, gamma);
    }

    private trainLinear(
        X: number[][], y: number[],
        C: number, epsilon: number, features: number
    ): ModelState {
        const n = X.length;
        const lr = 0.001;
        const epochs = 1000;

        const weights = new Array(features).fill(0);
        let bias = 0;

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < n; i++) {
                const pred = X[i].reduce((s, x, j) => s + weights[j] * x, 0) + bias;
                const error = y[i] - pred;

                if (Math.abs(error) > epsilon) {
                    const sign = error > 0 ? 1 : -1;
                    for (let j = 0; j < features; j++) {
                        weights[j] += lr * (C * sign * X[i][j] - weights[j] / n);
                    }
                    bias += lr * C * sign;
                }
            }
        }

        return {
            modelType: 'SVM',
            weights,
            bias,
            params: { C, epsilon, kernel: 0, gamma: 0 }
        };
    }

    private trainRBF(
        X: number[][], y: number[],
        C: number, epsilon: number, gamma: number
    ): ModelState {
        const n = X.length;
        // Simplified: use all training points as support vectors with weighted averaging
        const alphas = new Array(n).fill(0);
        const lr = 0.01;
        const epochs = 500;

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < n; i++) {
                let pred = 0;
                for (let j = 0; j < n; j++) {
                    pred += alphas[j] * this.rbfKernel(X[i], X[j], gamma);
                }
                const error = y[i] - pred;

                if (Math.abs(error) > epsilon) {
                    alphas[i] += lr * error;
                    alphas[i] = Math.max(-C, Math.min(C, alphas[i]));
                }
            }
        }

        return {
            modelType: 'SVM',
            supportVectors: X,
            weights: alphas,
            bias: 0,
            trainingData: { X, y },
            params: { C, epsilon, kernel: 1, gamma }
        };
    }

    predict(state: ModelState, X: number[][]): number[] {
        if (state.params.kernel === 0) {
            // Linear
            return X.map(row => {
                return row.reduce((s, x, j) => s + state.weights![j] * x, 0) + (state.bias || 0);
            });
        }

        // RBF
        const gamma = state.params.gamma;
        const sv = state.supportVectors!;
        const alphas = state.weights!;

        return X.map(row => {
            let pred = 0;
            for (let j = 0; j < sv.length; j++) {
                pred += alphas[j] * this.rbfKernel(row, sv[j], gamma);
            }
            return pred;
        });
    }

    private rbfKernel(a: number[], b: number[], gamma: number): number {
        const dist = a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
        return Math.exp(-gamma * dist);
    }
}
