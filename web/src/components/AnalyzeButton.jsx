export default function AnalyzeButton({ hasImage, hasNutrients, loading, onAnalyze }) {
  const disabled = loading || (!hasImage && !hasNutrients);

  let label = 'Analyze Soil';
  if (hasImage && !hasNutrients) label = 'Classify Soil Type';
  else if (!hasImage && hasNutrients) label = 'Predict Fertility';

  return (
    <button
      onClick={onAnalyze}
      disabled={disabled}
      className="w-full py-3 px-4 bg-emerald-600 text-white font-medium rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
    >
      {loading && (
        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {loading ? 'Analyzing...' : label}
    </button>
  );
}
