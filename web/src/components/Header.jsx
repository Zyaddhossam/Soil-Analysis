export default function Header() {
  return (
    <header className="bg-emerald-700 text-white py-4 px-6 shadow-md">
      <div className="max-w-6xl mx-auto flex items-center gap-3">
        <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <div>
          <h1 className="text-xl font-bold">Soil Analysis</h1>
          <p className="text-emerald-200 text-xs">Classify soil type and predict fertility</p>
        </div>
      </div>
    </header>
  );
}
