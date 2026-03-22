import { useState, useCallback } from 'react';
import {
    ScatterChart, Scatter, LineChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart
} from 'recharts';
import { calculateModel, saveExperiment, getExperiments, deleteExperiment } from '../services/api';
import type { DataPoint, RegressionResult, Experiment } from '../services/api';

export default function PredictionsPage() {
    // State
    const [experimentType, setExperimentType] = useState<'LUZ' | 'SOMBRA'>('LUZ');
    const [degree, setDegree] = useState(2);
    const [dataRows, setDataRows] = useState<{ x: string; y: string }[]>([
        { x: '', y: '' }, { x: '', y: '' }, { x: '', y: '' }
    ]);
    const [result, setResult] = useState<RegressionResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [historyOpen, setHistoryOpen] = useState(false);
    const [history, setHistory] = useState<Experiment[]>([]);

    // Simulation state
    const [simTime, setSimTime] = useState(12);
    const [simSpeed, setSimSpeed] = useState(5);
    const [simRunning, setSimRunning] = useState(false);

    // Save dialog
    const [saveOpen, setSaveOpen] = useState(false);
    const [saveName, setSaveName] = useState('');

    const addRow = () => setDataRows(prev => [...prev, { x: '', y: '' }]);

    const removeRow = (idx: number) => {
        if (dataRows.length <= 2) return;
        setDataRows(prev => prev.filter((_, i) => i !== idx));
    };

    const updateRow = (idx: number, field: 'x' | 'y', value: string) => {
        setDataRows(prev => prev.map((row, i) => i === idx ? { ...row, [field]: value } : row));
    };

    const handleCalculate = useCallback(async () => {
        setError(null);
        const data: DataPoint[] = dataRows
            .filter(r => r.x !== '' && r.y !== '')
            .map(r => ({ x: parseFloat(r.x), y: parseFloat(r.y) }));

        if (data.length < 2) {
            setError('Ingrese al menos 2 puntos de datos válidos.');
            return;
        }

        if (data.some(p => isNaN(p.x) || isNaN(p.y))) {
            setError('Todos los valores deben ser numéricos.');
            return;
        }

        setLoading(true);
        try {
            const res = await calculateModel(data, degree, experimentType);
            setResult(res);
        } catch (err: any) {
            setError(err.response?.data?.error || err.message);
        } finally {
            setLoading(false);
        }
    }, [dataRows, degree, experimentType]);

    const handleSave = async () => {
        if (!result || !saveName.trim()) return;
        const data = dataRows.filter(r => r.x !== '' && r.y !== '').map(r => ({ x: parseFloat(r.x), y: parseFloat(r.y) }));
        try {
            await saveExperiment({
                name: saveName, type: experimentType, degree,
                coefficients: result.coefficients, r2: result.r2, mse: result.mse,
                dataPoints: data
            });
            setSaveOpen(false);
            setSaveName('');
        } catch (e) { /* silent */ }
    };

    const openHistory = async () => {
        try {
            const data = await getExperiments();
            setHistory(data);
        } catch (e) { /* silent */ }
        setHistoryOpen(true);
    };

    const loadExperiment = (exp: Experiment) => {
        setExperimentType(exp.type);
        setDegree(exp.degree);
        setDataRows(exp.dataPoints.map(p => ({ x: String(p.x), y: String(p.y) })));
        setHistoryOpen(false);
        // Auto-calculate after loading
        setTimeout(() => handleCalculate(), 100);
    };

    const yLabel = experimentType === 'LUZ' ? 'Intensidad (Y)' : 'Long. Sombra (Y)';
    const chartTitle = experimentType === 'LUZ' ? 'Ajuste de Intensidad Solar' : 'Ajuste de Longitud de Sombra';

    // Merge scatter and line data for composed chart
    const scatterData = dataRows
        .filter(r => r.x !== '' && r.y !== '')
        .map(r => ({ x: parseFloat(r.x), y: parseFloat(r.y) }));

    return (
        <div className="flex h-full overflow-hidden">
            {/* LEFT PANEL: Controls */}
            <aside className="no-print w-[380px] min-w-[340px] border-r border-slate-200 bg-white flex flex-col overflow-hidden shrink-0">
                <div className="flex-1 overflow-y-auto p-6 custom-scrollbar space-y-8">
                    {/* Experiment Mode Toggle */}
                    <section>
                        <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
                            Modo de Experimento
                        </label>
                        <div className="bg-slate-100 p-1 rounded-xl flex">
                            <button
                                onClick={() => setExperimentType('LUZ')}
                                className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-semibold rounded-lg transition-all ${experimentType === 'LUZ'
                                        ? 'bg-white shadow-sm text-slate-900'
                                        : 'text-slate-500 hover:text-slate-700'
                                    }`}
                            >
                                <span className={experimentType === 'LUZ' ? '' : 'grayscale opacity-50'}>☀️</span> Modo Luz
                            </button>
                            <button
                                onClick={() => setExperimentType('SOMBRA')}
                                className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-semibold rounded-lg transition-all ${experimentType === 'SOMBRA'
                                        ? 'bg-white shadow-sm text-slate-900'
                                        : 'text-slate-500 hover:text-slate-700'
                                    }`}
                            >
                                <span className={experimentType === 'SOMBRA' ? '' : 'grayscale opacity-50'}>📏</span> Modo Sombra
                            </button>
                        </div>
                    </section>

                    {/* Polynomial Degree Slider */}
                    <section>
                        <div className="flex justify-between items-end mb-3">
                            <label className="block text-xs font-bold uppercase tracking-wider text-slate-500">
                                Grado Polinomial
                            </label>
                            <span className="text-3xl font-mono font-bold text-primary-600 leading-none">n={degree}</span>
                        </div>
                        <input
                            type="range" min="1" max="6" value={degree}
                            onChange={e => setDegree(parseInt(e.target.value))}
                            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                        />
                        <div className="flex justify-between text-[10px] text-slate-400 mt-2 font-mono">
                            {[1, 2, 3, 4, 5, 6].map(n => <span key={n}>{n}</span>)}
                        </div>
                    </section>

                    {/* Data Entry Table */}
                    <section className="space-y-4">
                        <div className="flex justify-between items-center">
                            <label className="block text-xs font-bold uppercase tracking-wider text-slate-500">
                                Datos de Medición
                            </label>
                            <span className="text-[10px] text-slate-400 uppercase">
                                Total: {dataRows.filter(r => r.x && r.y).length}
                            </span>
                        </div>

                        <div className="space-y-2">
                            {/* Header */}
                            <div className="grid grid-cols-12 gap-2 px-2 text-[10px] font-bold text-slate-400 uppercase">
                                <div className="col-span-5">Hora (X)</div>
                                <div className="col-span-5">{yLabel}</div>
                                <div className="col-span-2"></div>
                            </div>

                            {/* Rows */}
                            {dataRows.map((row, idx) => (
                                <div key={idx} className="grid grid-cols-12 gap-2 items-center">
                                    <div className="col-span-5">
                                        <input
                                            type="number" step="0.1" value={row.x}
                                            onChange={e => updateRow(idx, 'x', e.target.value)}
                                            placeholder="ej. 8.0"
                                            className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all outline-none"
                                        />
                                    </div>
                                    <div className="col-span-5">
                                        <input
                                            type="number" step="0.1" value={row.y}
                                            onChange={e => updateRow(idx, 'y', e.target.value)}
                                            placeholder="ej. 150"
                                            className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all outline-none"
                                        />
                                    </div>
                                    <div className="col-span-2 flex justify-center">
                                        <button onClick={() => removeRow(idx)} className="text-slate-300 hover:text-rose-500 transition-colors">
                                            <svg fill="none" height="18" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" width="18">
                                                <path d="M3 6h18" /><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                                                <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <button
                            onClick={addRow}
                            className="w-full py-3 border-2 border-dashed border-slate-200 rounded-xl text-slate-400 text-sm font-medium hover:bg-slate-50 hover:border-primary-300 hover:text-primary-500 transition-all flex items-center justify-center gap-2"
                        >
                            <svg fill="none" height="18" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" width="18">
                                <path d="M5 12h14" /><path d="M12 5v14" />
                            </svg>
                            Añadir Medición
                        </button>
                    </section>

                    {/* Error display */}
                    {error && (
                        <div className="p-3 bg-rose-50 text-rose-700 rounded-xl text-sm border border-rose-100">
                            {error}
                        </div>
                    )}
                </div>

                {/* Sticky Action Area */}
                <div className="p-6 border-t border-slate-100 bg-white/80 backdrop-blur-md space-y-3">
                    <button
                        onClick={handleCalculate}
                        disabled={loading}
                        className="w-full py-4 bg-primary-600 hover:bg-primary-700 disabled:opacity-60 text-white font-bold rounded-2xl shadow-lg shadow-primary-200 transition-all active:scale-[0.98] flex items-center justify-center gap-3"
                    >
                        {loading ? (
                            <span className="animate-spin">⏳</span>
                        ) : (
                            <svg fill="none" height="20" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" width="20">
                                <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
                            </svg>
                        )}
                        Calcular Modelo
                    </button>

                    <div className="flex gap-2">
                        <button
                            onClick={openHistory}
                            className="flex-1 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-100 rounded-xl transition-colors"
                        >
                            Historial
                        </button>
                        <button
                            onClick={() => window.print()}
                            className="flex-1 py-2.5 bg-primary-600 text-white text-sm font-medium rounded-xl hover:bg-primary-700 transition-shadow shadow-md flex items-center justify-center gap-2"
                        >
                            <svg fill="none" height="16" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" width="16">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7 10 12 15 17 10" />
                                <line x1="12" x2="12" y1="15" y2="3" />
                            </svg>
                            Exportar PDF
                        </button>
                    </div>
                </div>
            </aside>

            {/* RIGHT PANEL: Visualization & Analytics */}
            <section className="flex-1 h-full overflow-y-auto custom-scrollbar p-8 print-full-width">
                {/* Metrics Grid */}
                {result && (
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8 animate-in">
                        {/* Main Equation Card */}
                        <div className="md:col-span-2 p-6 rounded-2xl bg-primary-600 text-white shadow-xl flex flex-col justify-between">
                            <div className="flex items-center justify-between mb-4">
                                <span className="text-xs font-bold uppercase tracking-widest text-primary-200">Modelo Matemático</span>
                                <svg fill="none" height="20" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" width="20" className="text-primary-200">
                                    <path d="M16 4h4v4" /><path d="M12 4v1a2 2 0 0 0 2 2h2a2 2 0 0 1 2 2v1" />
                                    <path d="M8 4H4v4" /><path d="M12 4v1a2 2 0 0 1-2 2H8a2 2 0 0 0-2 2v1" />
                                    <path d="M16 20h4v-4" /><path d="M12 20v-1a2 2 0 0 1 2-2h2a2 2 0 0 0 2-2v-1" />
                                    <path d="M8 20H4v-4" /><path d="M12 20v-1a2 2 0 0 0-2-2H8a2 2 0 0 1-2-2v-1" />
                                </svg>
                            </div>
                            <div className="font-mono text-xl font-medium tracking-tight overflow-x-auto whitespace-nowrap">
                                {result.equation}
                            </div>
                        </div>

                        {/* R² Card */}
                        <div className="p-6 rounded-2xl bg-white border border-emerald-100 shadow-md flex flex-col justify-between">
                            <span className="text-xs font-bold uppercase tracking-widest text-slate-400">R² (Precisión)</span>
                            <div className="text-3xl font-bold text-emerald-500 mt-2">{result.r2.toFixed(4)}</div>
                            <div className="w-full bg-emerald-50 h-1.5 rounded-full mt-2 overflow-hidden">
                                <div className="bg-emerald-500 h-full transition-all duration-500" style={{ width: `${Math.max(0, result.r2 * 100)}%` }} />
                            </div>
                        </div>

                        {/* MSE Card */}
                        <div className="p-6 rounded-2xl bg-white border border-rose-100 shadow-md flex flex-col justify-between">
                            <span className="text-xs font-bold uppercase tracking-widest text-slate-400">MSE (Error)</span>
                            <div className="text-3xl font-bold text-rose-500 mt-2">{result.mse.toFixed(4)}</div>
                            <div className="w-full bg-rose-50 h-1.5 rounded-full mt-2 overflow-hidden">
                                <div className="bg-rose-500 h-full transition-all duration-500" style={{ width: `${Math.min(100, result.mse * 10)}%` }} />
                            </div>
                        </div>
                    </div>
                )}

                {/* Main Plot Area */}
                {result && (
                    <div className="bg-white rounded-3xl p-8 shadow-sm border border-slate-200 mb-8">
                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h3 className="text-lg font-bold text-slate-900">{chartTitle}</h3>
                                <p className="text-sm text-slate-500">Regresión polinomial de grado {degree}</p>
                            </div>
                            <div className="flex items-center gap-6">
                                <div className="flex items-center gap-2">
                                    <span className="w-3 h-3 rounded-full bg-amber-500" />
                                    <span className="text-xs font-medium text-slate-600">Mediciones</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="w-6 h-1 rounded-full bg-primary-600" />
                                    <span className="text-xs font-medium text-slate-600">Modelo</span>
                                </div>
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height={400}>
                            <ComposedChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis
                                    dataKey="x" type="number"
                                    label={{ value: 'Hora', position: 'bottom', offset: 0, fill: '#94a3b8', fontSize: 12 }}
                                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                                />
                                <YAxis
                                    label={{ value: experimentType === 'LUZ' ? 'Lux' : 'cm', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }}
                                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                                />
                                <Tooltip
                                    contentStyle={{ borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                />
                                <Legend />
                                <Line
                                    data={result.fittedPoints}
                                    dataKey="y" type="monotone" name="Modelo"
                                    stroke="#4f46e5" strokeWidth={3} dot={false}
                                />
                                <Scatter
                                    data={scatterData}
                                    dataKey="y" name="Mediciones"
                                    fill="#f59e0b" r={6}
                                />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* Derivative Chart (Shadow Mode Only) */}
                {result?.derivative && experimentType === 'SOMBRA' && (
                    <div className="bg-slate-900 rounded-3xl p-8 shadow-xl border border-slate-800 mb-8">
                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h3 className="text-lg font-bold text-white">Velocidad de Variación</h3>
                                <p className="text-sm text-slate-400">Primera derivada v(x) = dy/dx</p>
                            </div>
                            <div className="font-mono text-primary-400 bg-primary-500/10 px-4 py-1 rounded-full text-sm">
                                {result.derivative.equation}
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={result.derivative.points} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="x" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', color: '#e2e8f0' }} />
                                <Line dataKey="y" type="monotone" stroke="#818cf8" strokeWidth={3} dot={false} name="Velocidad" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* Empty state */}
                {!result && (
                    <div className="flex flex-col items-center justify-center h-full text-center opacity-50">
                        <svg fill="none" height="80" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" viewBox="0 0 24 24" width="80" className="text-slate-300 mb-4">
                            <circle cx="12" cy="12" r="4" />
                            <path d="M12 2v2" /><path d="M12 20v2" />
                            <path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" />
                            <path d="M2 12h2" /><path d="M20 12h2" />
                            <path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" />
                        </svg>
                        <h3 className="text-xl font-bold text-slate-400 mb-2">Ingresa datos y calcula</h3>
                        <p className="text-sm text-slate-400 max-w-md">
                            Introduce tus mediciones en la tabla de la izquierda, selecciona el grado polinomial y presiona "Calcular Modelo" para ver los resultados.
                        </p>
                    </div>
                )}
            </section>

            {/* History Sidebar */}
            <aside
                className={`fixed top-0 right-0 w-96 h-full bg-white shadow-2xl z-50 transform transition-transform duration-300 ease-in-out no-print ${historyOpen ? 'translate-x-0' : 'translate-x-full'
                    }`}
            >
                <div className="p-6 h-full flex flex-col">
                    <div className="flex items-center justify-between mb-8">
                        <h2 className="text-xl font-bold">Historial</h2>
                        <button onClick={() => setHistoryOpen(false)} className="p-2 hover:bg-slate-100 rounded-full">
                            <svg fill="none" height="20" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" width="20">
                                <path d="M18 6 6 18" /><path d="m6 6 12 12" />
                            </svg>
                        </button>
                    </div>
                    <div className="flex-1 space-y-4 overflow-y-auto custom-scrollbar pr-2">
                        {history.length === 0 && (
                            <p className="text-sm text-slate-400 text-center py-8">No hay experimentos guardados.</p>
                        )}
                        {history.map(exp => (
                            <div
                                key={exp.id}
                                onClick={() => loadExperiment(exp)}
                                className="p-4 rounded-2xl bg-slate-50 border border-slate-100 hover:border-primary-200 cursor-pointer group transition-all"
                            >
                                <div className="flex justify-between items-start mb-2">
                                    <span className={`px-2 py-0.5 text-[10px] font-bold rounded uppercase tracking-wider ${exp.type === 'LUZ'
                                            ? 'bg-amber-100 text-amber-700'
                                            : 'bg-primary-100 text-primary-700'
                                        }`}>
                                        {exp.type}
                                    </span>
                                    <span className="text-[10px] text-slate-400 font-medium">
                                        {new Date(exp.createdAt).toLocaleDateString('es-ES', { day: '2-digit', month: 'short', year: 'numeric' })}
                                    </span>
                                </div>
                                <p className="text-sm font-bold text-slate-800 mb-1">{exp.name}</p>
                                <div className="flex items-center gap-4 mt-3">
                                    <div className="text-[10px] text-slate-500">
                                        Grado: <span className="text-slate-900 font-bold">{exp.degree}</span>
                                    </div>
                                    {exp.r2 !== null && (
                                        <div className="text-[10px] text-slate-500">
                                            R²: <span className="text-emerald-500 font-bold">{exp.r2.toFixed(2)}</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </aside>

            {/* Save Dialog */}
            {saveOpen && (
                <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center" onClick={() => setSaveOpen(false)}>
                    <div className="bg-white rounded-2xl p-6 w-96 shadow-2xl" onClick={e => e.stopPropagation()}>
                        <h3 className="text-lg font-bold mb-4">Guardar Experimento</h3>
                        <input
                            type="text" value={saveName} onChange={e => setSaveName(e.target.value)}
                            placeholder="Nombre del experimento..."
                            className="w-full px-4 py-3 border border-slate-200 rounded-xl mb-4 focus:ring-2 focus:ring-primary-500 outline-none"
                        />
                        <div className="flex gap-3">
                            <button onClick={() => setSaveOpen(false)} className="flex-1 py-2.5 text-sm text-slate-600 hover:bg-slate-100 rounded-xl">
                                Cancelar
                            </button>
                            <button onClick={handleSave} className="flex-1 py-2.5 bg-primary-600 text-white text-sm font-medium rounded-xl">
                                Guardar
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
