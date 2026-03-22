// ============================================================
// Base Model — Strategy Pattern Interface
// ============================================================

export interface TrainableModel {
    /** Unique name of the algorithm */
    readonly name: string;
    /** Category: 'regression' | 'classification' */
    readonly category: string;
    /** Description for UI display */
    readonly description: string;
    /** Available hyperparameters with defaults */
    readonly hyperparameterSchema: HyperparameterDef[];

    /**
     * Train the model on the given data.
     * @param X - Input features (each row is a sample, each column a feature)
     * @param y - Target values
     * @param params - Hyperparameters
     * @returns Trained model state
     */
    train(X: number[][], y: number[], params: Record<string, number>): ModelState;

    /**
     * Predict using a trained model state.
     * @param state - Previously trained model state
     * @param X - Input features to predict
     * @returns Predicted values
     */
    predict(state: ModelState, X: number[][]): number[];
}

export interface HyperparameterDef {
    name: string;
    label: string;
    type: 'int' | 'float' | 'select';
    min?: number;
    max?: number;
    step?: number;
    default: number;
    options?: { value: number; label: string }[];
}

export interface ModelState {
    modelType: string;
    coefficients?: number[];
    tree?: DecisionNode;
    supportVectors?: number[][];
    weights?: number[];
    bias?: number;
    trainingData?: { X: number[][]; y: number[] };
    params: Record<string, number>;
}

export interface DecisionNode {
    featureIndex?: number;
    threshold?: number;
    value?: number; // leaf prediction
    left?: DecisionNode;
    right?: DecisionNode;
}

export interface TrainingMetrics {
    r2: number;
    mse: number;
    mae: number;
    rmse: number;
    predictions: number[];
    residuals: number[];
}
