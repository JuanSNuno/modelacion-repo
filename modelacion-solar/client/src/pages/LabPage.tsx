import { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter,
    Legend, ComposedChart
} from 'recharts';
import { getModelCatalog, trainModel, compareModels } from '../services/api';
import type { ModelInfo, TrainResult } from '../services/api';

const MODEL_ICONS: Record<string, string> = {
    LINEAR: '📈',
    POLYNOMIAL: '📊',
    DECISION_TREE: '🌳',
    SVM: '🎯',
    KNN: '🔍'
};

const MODEL_COLORS: Record<string, string> = {
    LINEAR: '#4f46e5',
    POLYNOMIAL: '#06b6d4',
    DECISION_TREE: '#10b981',
    SVM: '#f59e0b',
    KNN: '#ef4444'
};

export default function LabPage() {
    const [catalog, setCatalog] = useState<ModelInfo[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('LINEAR');
    const [params, setParams] = useState<Record<string, number>>({});
    const [dataInput, setDataInput] = useState<string>(
        '// Formato: x,y (un punto por línea)\n6,180\n8,420\n10,750\n12,980\n14,850\n16,520\n18,200'
    );
    const [result, setResult] = useState<TrainResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Compare mode
    const [compareMode, setCompareMode] = useState(false);
    const [compareResults, setCompareResults] = useState<any[]>([]);
    const [selectedModelsForCompare, setSelectedModelsForCompare] = useState<string[]>(['LINEAR', 'POLYNOMIAL']);

    useEffect(() => {
        getModelCatalog().then(cat => {
            setCatalog(cat);
            if (cat.length > 0) {
                const first = cat[0];
                setSelectedModel(first.name);
                const defaultParams: Record<string, number> = {};
                first.hyperparameters.forEach(h => { defaultParams[h.name] = h.default; });
                setParams(defaultParams);
            }
        }).catch(() => { });
    }, []);

    const selectModel = (name: string) => {
        setSelectedModel(name);
        const model = catalog.find(m => m.name === name);
        if (model) {
            const defaultParams: Record<string, number> = {};
            model.hyperparameters.forEach(h => { defaultParams[h.name] = h.default; });
            setParams(defaultParams);
        }
        setResult(null);
    };

    const parseData = (): { X: number[][]; y: number[] } | null => {
        const lines = dataInput.split('\n').filter(l => !l.startsWith('//') && l.trim());
        const X: number[][] = [];
        const y: number[] = [];
        for (const line of lines) {
            const parts = line.split(',').map(s => parseFloat(s.trim()));
            if (parts.length < 2 || parts.some(isNaN)) continue;
            X.push([parts[0]]);
            y.push(parts[1]);
        }
        if (X.length < 2) return null;
        return { X, y };
    };

    const handleTrain = async () => {
        setError(null);
        const parsed = parseData();
        if (!parsed) {
            setError('Se necesitan al menos 2 puntos válidos. Formato: x,y');
            return;
        }
        setLoading(true);
        try {
            const res = await trainModel(selectedModel, parsed.X, parsed.y, params);
            setResult(res);
        } catch (err: any) {
            setError(err.response?.data?.error || err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleCompare = async () => {
        setError(null);
        const parsed = parseData();
        if (!parsed) {
            setError('Se necesitan datos para comparar modelos.');
            return;
        }

        const modelsToCompare = selectedModelsForCompare.map(name => {
            const model = catalog.find(m => m.name === name);
            const p: Record<string, number> = {};
            model?.hyperparameters.forEach(h => { p[h.name] = h.default; });
            return { name, params: p };
        });

        setLoading(true);
        try {
            const res = await compareModels(modelsToCompare, parsed.X, parsed.y);
            setCompareResults(res);
        } catch (err: any) {
            setError(err.response?.data?.error || err.message);
        } finally {
            setLoading(false);
        }
    };

    const toggleCompareModel = (name: string) => {
        setSelectedModelsForCompare(prev =>
            prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
        );
    };

    const currentModel = catalog.find(m => m.name === selectedModel);

    // Prepare chart data
    const parsed = parseData();
    const chartData = parsed ? parsed.X.map((x, i) => ({
        x: x[0],
        actual: parsed.y[i],
        predicted: result?.predictions?.[i] ?? null,
        residual: result?.residuals?.[i] ?? null
    })) : [];

    return (
        <div className="flex h-full overflow-hidden">
            {/* LEFT PANEL: Model Selection + Config */}
            <aside className="w-[380px] min-w-[340px] border-r border-slate-200 bg-white flex flex-col overflow-hidden shrink-0">
                <div className="flex-1 overflow-y-auto p-6 custom-scrollbar space-y-6">
                    {/* Lab Header */}
                    <div>
                        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                            <span className="text-2xl">🧪</span> Laboratorio ML
                        </h2>
                        <p className="text-xs text-slate-500 mt-1">Entrena, evalúa y compara modelos de Machine Learning</p>
                    </div>

                    {/* Mode Toggle */}
                    <div className="bg-slate-100 p-1 rounded-xl flex">
                        <button
                            onClick={() => setCompareMode(false)}
                            className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all ${!compareMode ? 'bg-white shadow-sm text-slate-900' : 'text-slate-500'
                                }`}
                        >
                            Entrenar
                        </button>
                        <button
                            onClick={() => setCompareMode(true)}
                            className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all ${compareMode ? 'bg-white shadow-sm text-slate-900' : 'text-slate-500'
                                }`}
                        >
                            Comparar
                        </button>
                    </div>

                    {!compareMode ? (
                        <>
                            {/* Model Selector Cards */}
                            <section>
                                <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
                                    Seleccionar Modelo
                                </label>
                                <div className="space-y-2">
                                    {catalog.map(model => (
                                        <button
                                            key={model.name}
                                            onClick={() => selectModel(model.name)}
                                            className={`w-full text-left p-3 rounded-xl border transition-all ${selectedModel === model.name
                                                    ? 'border-primary-300 bg-primary-50 ring-2 ring-primary-200'
                                                    : 'border-slate-100 bg-slate-50 hover:border-primary-200'
                                                }`}
                                        >
                                            <div className="flex items-center gap-3">
                                                <span className="text-xl">{MODEL_ICONS[model.name] || '📦'}</span>
                                                <div>
                                                    <div className="text-sm font-bold text-slate-800">{model.name.replace(/_/g, ' ')}</div>
                                                    <div className="text-[10px] text-slate-500 line-clamp-1">{model.description}</div>
                                                </div>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </section>

                            {/* Hyperparameters */}
                            {currentModel && currentModel.hyperparameters.length > 0 && (
                                <section>
                                    <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
                                        Hiperparámetros
                                    </label>
                                    <div className="space-y-4">
                                        {currentModel.hyperparameters.map(hp => (
                                            <div key={hp.name}>
                                                <div className="flex justify-between items-center mb-1">
                                                    <span className="text-xs font-medium text-slate-600">{hp.label}</span>
                                                    <span className="text-xs font-mono font-bold text-primary-600">
                                                        {hp.type === 'select'
                                                            ? hp.options?.find(o => o.value === (params[hp.name] ?? hp.default))?.label
                                                            : (params[hp.name] ?? hp.default)
                                                        }
                                                    </span>
                                                </div>
                                                {hp.type === 'select' ? (
                                                    <select
                                                        value={params[hp.name] ?? hp.default}
                                                        onChange={e => setParams(p => ({ ...p, [hp.name]: parseFloat(e.target.value) }))}
                                                        className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 outline-none"
                                                    >
                                                        {hp.options?.map(opt => (
                                                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                                                        ))}
                                                    </select>
                                                ) : (
                                                    <input
                                                        type="range"
                                                        min={hp.min} max={hp.max} step={hp.step}
                                                        value={params[hp.name] ?? hp.default}
                                                        onChange={e => setParams(p => ({ ...p, [hp.name]: parseFloat(e.target.value) }))}
                                                        className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                                                    />
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </section>
                            )}
                        </>
                    ) : (
                        /* Compare Mode: Model Selection */
                        <section>
                            <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
                                Modelos a Comparar
                            </label>
                            <div className="space-y-2">
                                {catalog.map(model => (
                                    <label
                                        key={model.name}
                                        className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all ${selectedModelsForCompare.includes(model.name)
                                                ? 'border-primary-300 bg-primary-50'
                                                : 'border-slate-100 bg-slate-50'
                                            }`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={selectedModelsForCompare.includes(model.name)}
                                            onChange={() => toggleCompareModel(model.name)}
                                            className="accent-primary-600"
                                        />
                                        <span className="text-xl">{MODEL_ICONS[model.name] || '📦'}</span>
                                        <span className="text-sm font-bold text-slate-800">{model.name.replace(/_/g, ' ')}</span>
                                    </label>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Data Input */}
                    <section>
                        <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
                            Datos de Entrenamiento
                        </label>
                        <textarea
                            value={dataInput}
                            onChange={e => setDataInput(e.target.value)}
                            rows={8}
                            className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl text-sm font-mono focus:ring-2 focus:ring-primary-500 outline-none resize-none"
                            placeholder="x,y (un punto por línea)"
                        />
                        <p className="text-[10px] text-slate-400 mt-1">
                            Puntos válidos: {parseData()?.X.length || 0}
                        </p>
                    </section>

                    {error && (
                        <div className="p-3 bg-rose-50 text-rose-700 rounded-xl text-sm border border-rose-100">
                            {error}
                        </div>
                    )}
                </div>

                {/* Action Button */}
                <div className="p-6 border-t border-slate-100 bg-white/80 backdrop-blur-md">
                    <button
                        onClick={compareMode ? handleCompare : handleTrain}
                        disabled={loading}
                        className="w-full py-4 bg-primary-600 hover:bg-primary-700 disabled:opacity-60 text-white font-bold rounded-2xl shadow-lg shadow-primary-200 transition-all active:scale-[0.98] flex items-center justify-center gap-3"
                    >
                        {loading ? <span className="animate-spin">⏳</span> : '🧠'}
                        {compareMode ? 'Comparar Modelos' : 'Entrenar Modelo'}
                    </button>
                </div>
            </aside>

            {/* RIGHT PANEL: Results */}
            <section className="flex-1 h-full overflow-y-auto custom-scrollbar p-8">
                {!compareMode && result ? (
                    <>
                        {/* Metrics Cards */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                            <div className="p-5 rounded-2xl bg-white border border-emerald-100 shadow-md">
                                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">R²</span>
                                <div className="text-2xl font-bold text-emerald-500 mt-1">{result.metrics.r2.toFixed(4)}</div>
                                <div className="w-full bg-emerald-50 h-1.5 rounded-full mt-2 overflow-hidden">
                                    <div className="bg-emerald-500 h-full" style={{ width: `${Math.max(0, result.metrics.r2 * 100)}%` }} />
                                </div>
                            </div>
                            <div className="p-5 rounded-2xl bg-white border border-rose-100 shadow-md">
                                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">MSE</span>
                                <div className="text-2xl font-bold text-rose-500 mt-1">{result.metrics.mse.toFixed(4)}</div>
                            </div>
                            <div className="p-5 rounded-2xl bg-white border border-blue-100 shadow-md">
                                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">MAE</span>
                                <div className="text-2xl font-bold text-blue-500 mt-1">{result.metrics.mae.toFixed(4)}</div>
                            </div>
                            <div className="p-5 rounded-2xl bg-white border border-purple-100 shadow-md">
                                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">RMSE</span>
                                <div className="text-2xl font-bold text-purple-500 mt-1">{result.metrics.rmse.toFixed(4)}</div>
                            </div>
                        </div>

                        {/* Predictions vs Actual Chart */}
                        <div className="bg-white rounded-3xl p-8 shadow-sm border border-slate-200 mb-8">
                            <div className="mb-6">
                                <h3 className="text-lg font-bold text-slate-900">Predicciones vs. Valores Reales</h3>
                                <p className="text-sm text-slate-500">{selectedModel.replace(/_/g, ' ')} — Ajuste del modelo</p>
                            </div>
                            <ResponsiveContainer width="100%" height={350}>
                                <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                    <XAxis dataKey="x" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                    <Tooltip contentStyle={{ borderRadius: '12px', border: '1px solid #e2e8f0' }} />
                                    <Legend />
                                    <Scatter dataKey="actual" name="Datos Reales" fill="#f59e0b" r={6} />
                                    <Line dataKey="predicted" name="Predicción" stroke="#4f46e5" strokeWidth={3} dot={false} type="monotone" />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Residuals Chart */}
                        <div className="bg-slate-900 rounded-3xl p-8 shadow-xl border border-slate-800">
                            <div className="mb-6">
                                <h3 className="text-lg font-bold text-white">Gráfica de Residuales</h3>
                                <p className="text-sm text-slate-400">Diferencia entre valores reales y predichos</p>
                            </div>
                            <ResponsiveContainer width="100%" height={250}>
                                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                    <XAxis dataKey="x" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', color: '#e2e8f0' }} />
                                    <Bar dataKey="residual" name="Residual" fill="#818cf8" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </>
                ) : compareMode && compareResults.length > 0 ? (
                    <>
                        {/* Comparison Table */}
                        <div className="bg-white rounded-3xl p-8 shadow-sm border border-slate-200 mb-8">
                            <h3 className="text-lg font-bold text-slate-900 mb-6">Comparación de Modelos</h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="text-left border-b border-slate-200">
                                            <th className="pb-3 text-xs font-bold uppercase tracking-wider text-slate-500">Modelo</th>
                                            <th className="pb-3 text-xs font-bold uppercase tracking-wider text-slate-500">R²</th>
                                            <th className="pb-3 text-xs font-bold uppercase tracking-wider text-slate-500">MSE</th>
                                            <th className="pb-3 text-xs font-bold uppercase tracking-wider text-slate-500">MAE</th>
                                            <th className="pb-3 text-xs font-bold uppercase tracking-wider text-slate-500">RMSE</th>
                                            <th className="pb-3 text-xs font-bold uppercase tracking-wider text-slate-500">Estado</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {compareResults.map((r: any, i: number) => (
                                            <tr key={i} className="border-b border-slate-50 hover:bg-slate-50">
                                                <td className="py-3 font-bold flex items-center gap-2">
                                                    <span className="w-3 h-3 rounded-full" style={{ background: MODEL_COLORS[r.name] || '#94a3b8' }} />
                                                    {MODEL_ICONS[r.name] || '📦'} {r.name.replace(/_/g, ' ')}
                                                </td>
                                                <td className="py-3">
                                                    {r.metrics ? (
                                                        <span className={`font-mono font-bold ${r.metrics.r2 > 0.9 ? 'text-emerald-500' : r.metrics.r2 > 0.7 ? 'text-amber-500' : 'text-rose-500'}`}>
                                                            {r.metrics.r2.toFixed(4)}
                                                        </span>
                                                    ) : '—'}
                                                </td>
                                                <td className="py-3 font-mono">{r.metrics?.mse?.toFixed(4) || '—'}</td>
                                                <td className="py-3 font-mono">{r.metrics?.mae?.toFixed(4) || '—'}</td>
                                                <td className="py-3 font-mono">{r.metrics?.rmse?.toFixed(4) || '—'}</td>
                                                <td className="py-3">
                                                    {r.error ? (
                                                        <span className="text-xs text-rose-500 bg-rose-50 px-2 py-1 rounded">{r.error}</span>
                                                    ) : (
                                                        <span className="text-xs text-emerald-600 bg-emerald-50 px-2 py-1 rounded font-medium">✓ OK</span>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Comparison Bar Chart */}
                        <div className="bg-white rounded-3xl p-8 shadow-sm border border-slate-200">
                            <h3 className="text-lg font-bold text-slate-900 mb-6">R² por Modelo</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart
                                    data={compareResults.filter((r: any) => r.metrics).map((r: any) => ({
                                        name: r.name.replace(/_/g, ' '),
                                        r2: r.metrics.r2,
                                        mse: r.metrics.mse,
                                        fill: MODEL_COLORS[r.name] || '#94a3b8'
                                    }))}
                                    margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                    <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} angle={-15} textAnchor="end" />
                                    <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                    <Tooltip contentStyle={{ borderRadius: '12px', border: '1px solid #e2e8f0' }} />
                                    <Bar dataKey="r2" name="R²" radius={[8, 8, 0, 0]}>
                                        {compareResults.filter((r: any) => r.metrics).map((r: any, i: number) => (
                                            <rect key={i} fill={MODEL_COLORS[r.name] || '#94a3b8'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </>
                ) : (
                    /* Empty state */
                    <div className="flex flex-col items-center justify-center h-full text-center opacity-50">
                        <span className="text-6xl mb-4">🧪</span>
                        <h3 className="text-xl font-bold text-slate-400 mb-2">
                            {compareMode ? 'Selecciona modelos y compara' : 'Entrena un modelo'}
                        </h3>
                        <p className="text-sm text-slate-400 max-w-md">
                            {compareMode
                                ? 'Selecciona los modelos que deseas comparar, ingresa los datos y presiona "Comparar Modelos".'
                                : 'Elige un algoritmo, configura los hiperparámetros, ingresa los datos de entrenamiento y presiona "Entrenar Modelo".'}
                        </p>
                    </div>
                )}
            </section>
        </div>
    );
}
