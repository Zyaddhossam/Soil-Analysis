import SoilTypeResult from './SoilTypeResult';
import FertilityResult from './FertilityResult';

function LabTestBanner() {
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start gap-3">
      <svg className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <div>
        <p className="text-sm font-medium text-blue-800">Want a full analysis?</p>
        <p className="text-xs text-blue-600 mt-1">
          Enter soil lab values (N, P, K, pH, etc.) in the form to get
          fertility predictions alongside soil type classification.
        </p>
      </div>
    </div>
  );
}

export default function ResultsPanel({ results, loading }) {
  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2].map((i) => (
          <div key={i} className="bg-white border border-gray-200 rounded-lg p-5 animate-pulse space-y-3">
            <div className="flex justify-between">
              <div className="h-5 bg-gray-200 rounded w-24" />
              <div className="h-6 bg-gray-200 rounded-full w-28" />
            </div>
            <div className="h-3 bg-gray-200 rounded-full w-full" />
            <div className="space-y-2">
              <div className="h-3 bg-gray-200 rounded w-3/4" />
              <div className="h-3 bg-gray-200 rounded w-1/2" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (!results) {
    return (
      <div className="flex flex-col items-center justify-center text-center py-16 text-gray-400">
        <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <p className="text-sm">Upload an image or enter lab values to begin analysis</p>
      </div>
    );
  }

  const imageOnly = results.soilType && !results.fertility;

  return (
    <div className="space-y-4">
      {results.soilType && <SoilTypeResult data={results.soilType} />}
      {imageOnly && <LabTestBanner />}
      {results.fertility && <FertilityResult data={results.fertility} />}
    </div>
  );
}
