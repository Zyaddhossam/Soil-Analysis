import ConfidenceBar from './ConfidenceBar';
import ProbabilityChart from './ProbabilityChart';

const FERTILITY_COLORS = {
  'Less Fertile': 'bg-red-100 text-red-800',
  'Fertile': 'bg-amber-100 text-amber-800',
  'Highly Fertile': 'bg-green-100 text-green-800',
};

export default function FertilityResult({ data }) {
  const badgeClass = FERTILITY_COLORS[data.class_name] || 'bg-gray-100 text-gray-800';

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">Fertility</h3>
        <span className={`text-sm font-medium px-3 py-1 rounded-full ${badgeClass}`}>
          {data.class_name}
        </span>
      </div>

      <ConfidenceBar value={data.confidence} />

      {data.recommendation && (
        <div className="bg-emerald-50 border-l-4 border-emerald-400 p-3 rounded-r-md">
          <p className="text-sm font-medium text-emerald-800 mb-1">Recommendation</p>
          <p className="text-sm text-emerald-700">{data.recommendation}</p>
        </div>
      )}

      {data.needs_lab_test && (
        <div className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded-r-md">
          <p className="text-sm font-medium text-blue-800 mb-1">Lab Test Recommended</p>
          <p className="text-sm text-blue-700">
            For more accurate fertility analysis, consider getting a soil lab test
            with nutrient measurements (N, P, K, pH, etc.).
          </p>
        </div>
      )}

      {data.warnings && data.warnings.length > 0 && (
        <div className="space-y-1">
          {data.warnings.map((warning, i) => (
            <div key={i} className="flex items-start gap-2 text-xs text-amber-700 bg-amber-50 px-3 py-2 rounded-md">
              <svg className="w-3.5 h-3.5 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {warning}
            </div>
          ))}
        </div>
      )}

      {data.probabilities && (
        <ProbabilityChart probabilities={data.probabilities} highlightClass={data.class_name} />
      )}
    </div>
  );
}
