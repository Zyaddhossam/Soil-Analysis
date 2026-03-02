export default function ProbabilityChart({ probabilities, highlightClass }) {
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-gray-600">Class Probabilities</h4>
      {sorted.map(([name, prob]) => {
        const percent = (prob * 100).toFixed(1);
        const isHighlight = name === highlightClass;
        return (
          <div key={name} className="flex items-center gap-2">
            <span
              className={`text-xs w-28 shrink-0 text-right ${isHighlight ? 'font-bold text-gray-900' : 'text-gray-500'}`}
            >
              {name}
            </span>
            <div className="flex-1 bg-gray-100 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${isHighlight ? 'bg-emerald-500' : 'bg-gray-300'}`}
                style={{ width: `${percent}%` }}
              />
            </div>
            <span
              className={`text-xs w-12 ${isHighlight ? 'font-bold text-gray-900' : 'text-gray-500'}`}
            >
              {percent}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
