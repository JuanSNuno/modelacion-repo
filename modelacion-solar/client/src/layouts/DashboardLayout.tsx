import { Outlet, NavLink } from 'react-router-dom';

export default function DashboardLayout() {
    const linkClass = ({ isActive }: { isActive: boolean }) =>
        `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${isActive
            ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/30'
            : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800'
        }`;

    return (
        <div className="h-screen flex bg-slate-50 font-sans text-slate-800 overflow-hidden">
            {/* Sidebar Navigation */}
            <nav className="no-print w-20 lg:w-64 bg-white border-r border-slate-200 flex flex-col py-6 px-3 lg:px-5 shrink-0">
                {/* Logo */}
                <div className="flex items-center gap-2 mb-10 px-1">
                    <div className="w-10 h-10 bg-amber-100 rounded-xl flex items-center justify-center text-amber-500 shrink-0">
                        <svg fill="none" height="24" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" width="24">
                            <circle cx="12" cy="12" r="4" />
                            <path d="M12 2v2" /><path d="M12 20v2" />
                            <path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" />
                            <path d="M2 12h2" /><path d="M20 12h2" />
                            <path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" />
                        </svg>
                    </div>
                    <h1 className="text-xl font-bold tracking-tight text-slate-900 hidden lg:block">
                        Solar<span className="text-primary-600">Motion</span>
                    </h1>
                </div>

                {/* Nav Links */}
                <div className="flex flex-col gap-2 flex-1">
                    <NavLink to="/predictions" className={linkClass}>
                        <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
                        </svg>
                        <span className="hidden lg:inline">Predicciones</span>
                    </NavLink>

                    <NavLink to="/lab" className={linkClass}>
                        <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
                        </svg>
                        <span className="hidden lg:inline">Laboratorio ML</span>
                    </NavLink>
                </div>

                {/* Version */}
                <div className="text-[10px] text-slate-400 text-center hidden lg:block">
                    SolarMotion v1.0
                </div>
            </nav>

            {/* Main Content */}
            <main className="flex-1 overflow-hidden">
                <Outlet />
            </main>
        </div>
    );
}
