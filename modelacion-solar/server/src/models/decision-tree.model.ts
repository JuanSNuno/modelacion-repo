// ============================================================
// Decision Tree Regressor — CART Algorithm
// ============================================================

import { TrainableModel, ModelState, HyperparameterDef, DecisionNode } from './base.model';

export class DecisionTreeModel implements TrainableModel {
    readonly name = 'DECISION_TREE';
    readonly category = 'regression';
    readonly description = 'Árbol de Decisión (CART) — Regresión por particiones recursivas';
    readonly hyperparameterSchema: HyperparameterDef[] = [
        { name: 'maxDepth', label: 'Profundidad Máxima', type: 'int', min: 1, max: 20, step: 1, default: 5 },
        { name: 'minSamples', label: 'Muestras Mín. por Nodo', type: 'int', min: 1, max: 50, step: 1, default: 2 }
    ];

    train(X: number[][], y: number[], params: Record<string, number>): ModelState {
        const maxDepth = params.maxDepth || 5;
        const minSamples = params.minSamples || 2;

        const tree = this.buildTree(X, y, 0, maxDepth, minSamples);

        return {
            modelType: this.name,
            tree,
            params
        };
    }

    predict(state: ModelState, X: number[][]): number[] {
        return X.map(row => this.predictOne(state.tree!, row));
    }

    private buildTree(
        X: number[][], y: number[],
        depth: number, maxDepth: number, minSamples: number
    ): DecisionNode {
        const n = X.length;

        // Leaf conditions
        if (n <= minSamples || depth >= maxDepth || this.allSame(y)) {
            return { value: this.mean(y) };
        }

        const features = X[0]?.length || 1;
        let bestFeature = 0;
        let bestThreshold = 0;
        let bestMSE = Infinity;
        let bestLeftIdx: number[] = [];
        let bestRightIdx: number[] = [];

        // Try each feature and each possible split
        for (let f = 0; f < features; f++) {
            // Get unique sorted values for this feature
            const values = [...new Set(X.map(row => row[f]))].sort((a, b) => a - b);

            for (let i = 0; i < values.length - 1; i++) {
                const threshold = (values[i] + values[i + 1]) / 2;
                const leftIdx: number[] = [];
                const rightIdx: number[] = [];

                for (let j = 0; j < n; j++) {
                    if (X[j][f] <= threshold) {
                        leftIdx.push(j);
                    } else {
                        rightIdx.push(j);
                    }
                }

                if (leftIdx.length === 0 || rightIdx.length === 0) continue;

                const leftY = leftIdx.map(i => y[i]);
                const rightY = rightIdx.map(i => y[i]);
                const mse = (this.variance(leftY) * leftY.length + this.variance(rightY) * rightY.length) / n;

                if (mse < bestMSE) {
                    bestMSE = mse;
                    bestFeature = f;
                    bestThreshold = threshold;
                    bestLeftIdx = leftIdx;
                    bestRightIdx = rightIdx;
                }
            }
        }

        // If no valid split found, make leaf
        if (bestLeftIdx.length === 0 || bestRightIdx.length === 0) {
            return { value: this.mean(y) };
        }

        const leftX = bestLeftIdx.map(i => X[i]);
        const leftY = bestLeftIdx.map(i => y[i]);
        const rightX = bestRightIdx.map(i => X[i]);
        const rightY = bestRightIdx.map(i => y[i]);

        return {
            featureIndex: bestFeature,
            threshold: bestThreshold,
            left: this.buildTree(leftX, leftY, depth + 1, maxDepth, minSamples),
            right: this.buildTree(rightX, rightY, depth + 1, maxDepth, minSamples)
        };
    }

    private predictOne(node: DecisionNode, x: number[]): number {
        if (node.value !== undefined) return node.value;
        if (x[node.featureIndex!] <= node.threshold!) {
            return this.predictOne(node.left!, x);
        }
        return this.predictOne(node.right!, x);
    }

    private mean(arr: number[]): number {
        return arr.reduce((s, v) => s + v, 0) / arr.length;
    }

    private variance(arr: number[]): number {
        const m = this.mean(arr);
        return arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
    }

    private allSame(arr: number[]): boolean {
        return arr.every(v => Math.abs(v - arr[0]) < 1e-10);
    }
}
