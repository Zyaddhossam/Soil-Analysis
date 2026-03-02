export default function ConfidenceBar({ value, label = 'Confidence' }) {
  const percent = (value * 100).toFixed(1);
  let barColor = 'bg-green-500';
  if (value < 0.5) barColor = 'bg-red-500';
  else if (value < 0.75) barColor = 'bg-amber-500';

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-600 font-medium">{label}</span>
        <span className="font-semibold">{percent}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div
          className={`${barColor} h-3 rounded-full transition-all duration-500`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
