import { useState } from 'react';
import { NUTRIENT_FIELDS, SAMPLE_VALUES, EMPTY_NUTRIENTS } from '../constants/nutrients';

export default function NutrientForm({ nutrients, onNutrientsChange, disabled }) {
  const [expanded, setExpanded] = useState(false);

  function handleChange(key, value) {
    onNutrientsChange({ ...nutrients, [key]: value });
  }

  function isOutOfRange(field, value) {
    if (value === '' || value === null) return false;
    const num = parseFloat(value);
    if (isNaN(num)) return true;
    return num < field.min || num > field.max;
  }

  return (
    <div className="border border-gray-200 rounded-lg">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
      >
        <span className="text-sm font-medium text-gray-700">
          Lab Values (Optional)
        </span>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => onNutrientsChange({ ...SAMPLE_VALUES })}
              disabled={disabled}
              className="text-xs px-3 py-1 bg-emerald-50 text-emerald-700 rounded-md hover:bg-emerald-100 disabled:opacity-50"
            >
              Fill Sample Data
            </button>
            <button
              type="button"
              onClick={() => onNutrientsChange({ ...EMPTY_NUTRIENTS })}
              disabled={disabled}
              className="text-xs px-3 py-1 bg-gray-100 text-gray-600 rounded-md hover:bg-gray-200 disabled:opacity-50"
            >
              Clear All
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {NUTRIENT_FIELDS.map((field) => {
              const outOfRange = isOutOfRange(field, nutrients[field.key]);
              return (
                <div key={field.key}>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    {field.key}
                    {field.unit && (
                      <span className="text-gray-400 font-normal"> ({field.unit})</span>
                    )}
                  </label>
                  <input
                    type="number"
                    value={nutrients[field.key]}
                    onChange={(e) => handleChange(field.key, e.target.value)}
                    disabled={disabled}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    placeholder={`${field.min}-${field.max}`}
                    className={`w-full text-sm px-2 py-1.5 border rounded-md disabled:opacity-50 focus:outline-none focus:ring-1 ${
                      outOfRange
                        ? 'border-red-300 focus:ring-red-400'
                        : 'border-gray-300 focus:ring-emerald-400'
                    }`}
                  />
                  {outOfRange && (
                    <p className="text-xs text-red-500 mt-0.5">
                      Range: {field.min}-{field.max}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
