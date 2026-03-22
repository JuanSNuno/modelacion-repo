import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    headers: { 'Content-Type': 'application/json' }
});

// ========== Polynomial Regression (Predictions) ==========
export interface DataPoint { x: number; y: number; }

export interface RegressionResult {
    coefficients: number[];
    equation: string;
    r2: number;
    mse: number;
    fittedPoints: DataPoint[];
    derivative?: {
        coefficients: number[];
        equation: string;
        points: DataPoint[];
    };
}

export async function calculateModel(
    data: DataPoint[],
    degree: number,
    experimentType: 'LUZ' | 'SOMBRA'
): Promise<RegressionResult> {
    const res = await api.post('/model/calculate', { data, degree, experimentType });
    return res.data;
}

// ========== Experiments ==========
export interface Experiment {
    id: string;
    name: string;
    type: 'LUZ' | 'SOMBRA';
    degree: number;
    coefficients: number[];
    r2: number | null;
    mse: number | null;
    createdAt: string;
    dataPoints: DataPoint[];
}

export async function getExperiments(): Promise<Experiment[]> {
    const res = await api.get('/experiments');
    return res.data;
}

export async function saveExperiment(data: {
    name: string;
    type: 'LUZ' | 'SOMBRA';
    degree: number;
    coefficients: number[];
    r2: number | null;
    mse: number | null;
    dataPoints: DataPoint[];
}): Promise<Experiment> {
    const res = await api.post('/experiments', data);
    return res.data;
}

export async function deleteExperiment(id: string): Promise<void> {
    await api.delete(`/experiments/${id}`);
}

// ========== Lab / ML Models ==========
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

export interface ModelInfo {
    name: string;
    category: string;
    description: string;
    hyperparameters: HyperparameterDef[];
}

export interface TrainResult {
    modelType: string;
    metrics: { r2: number; mse: number; mae: number; rmse: number };
    predictions: number[];
    residuals: number[];
}

export async function getModelCatalog(): Promise<ModelInfo[]> {
    const res = await api.get('/lab/models');
    return res.data;
}

export async function trainModel(
    modelType: string,
    X: number[][],
    y: number[],
    params: Record<string, number>
): Promise<TrainResult> {
    const res = await api.post('/lab/train', { modelType, X, y, params });
    return res.data;
}

export async function compareModels(
    models: { name: string; params: Record<string, number> }[],
    X: number[][],
    y: number[]
) {
    const res = await api.post('/lab/compare', { models, X, y });
    return res.data;
}

export default api;
