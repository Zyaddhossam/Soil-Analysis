import ConfidenceBar from './ConfidenceBar';
import ProbabilityChart from './ProbabilityChart';

const SOIL_COLORS = {
  'Alluvial Soil': 'bg-amber-100 text-amber-800',
  'Black Soil': 'bg-gray-200 text-gray-800',
  'Clay Soil': 'bg-orange-100 text-orange-800',
  'Red Soil': 'bg-red-100 text-red-800',
};

const BACKBONE_LABELS = {
  efficientnet_b0: 'EfficientNet-B0',
  mobilenet_v2: 'MobileNetV2',
  xception: 'Xception',
};

export default function SoilTypeResult({ data }) {
  const badgeClass = SOIL_COLORS[data.class_name] || 'bg-gray-100 text-gray-800';
  const backboneLabel = BACKBONE_LABELS[data.backbone] || data.backbone;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">Soil Type</h3>
        <span className={`text-sm font-medium px-3 py-1 rounded-full ${badgeClass}`}>
          {data.class_name}
        </span>
      </div>

      <ConfidenceBar value={data.confidence} />

      {data.backbone && (
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          <span>Model: {backboneLabel}</span>
        </div>
      )}

      {data.characteristics && (
        <div className="space-y-2 text-sm">
          {data.characteristics.description && (
            <div>
              <span className="font-medium text-gray-700">Description: </span>
              <span className="text-gray-600">{data.characteristics.description}</span>
            </div>
          )}
          {data.characteristics.suitable_crops && (
            <div>
              <span className="font-medium text-gray-700">Suitable Crops: </span>
              <span className="text-gray-600">{data.characteristics.suitable_crops}</span>
            </div>
          )}
          {data.characteristics.characteristics && (
            <div>
              <span className="font-medium text-gray-700">Characteristics: </span>
              <span className="text-gray-600">{data.characteristics.characteristics}</span>
            </div>
          )}
        </div>
      )}

      {data.probabilities && (
        <ProbabilityChart probabilities={data.probabilities} highlightClass={data.class_name} />
      )}
    </div>
  );
}
