export const NUTRIENT_FIELDS = [
  { key: 'N',  label: 'Nitrogen',               unit: 'kg/ha', min: 0,   max: 500,  step: 1,    sample: 280   },
  { key: 'P',  label: 'Phosphorus',             unit: 'kg/ha', min: 0,   max: 200,  step: 1,    sample: 45    },
  { key: 'K',  label: 'Potassium',              unit: 'kg/ha', min: 0,   max: 1000, step: 1,    sample: 320   },
  { key: 'pH', label: 'pH',                     unit: '',      min: 3.5, max: 10.0, step: 0.1,  sample: 6.5   },
  { key: 'EC', label: 'Electrical Conductivity', unit: 'dS/m', min: 0,   max: 10,   step: 0.01, sample: 0.45  },
  { key: 'OC', label: 'Organic Carbon',         unit: '%',     min: 0,   max: 10,   step: 0.01, sample: 0.75  },
  { key: 'S',  label: 'Sulfur',                 unit: 'mg/kg', min: 0,   max: 100,  step: 1,    sample: 12    },
  { key: 'Zn', label: 'Zinc',                   unit: 'mg/kg', min: 0,   max: 50,   step: 0.1,  sample: 1.2   },
  { key: 'Fe', label: 'Iron',                   unit: 'mg/kg', min: 0,   max: 500,  step: 0.1,  sample: 8.5   },
  { key: 'Cu', label: 'Copper',                 unit: 'mg/kg', min: 0,   max: 50,   step: 0.1,  sample: 1.8   },
  { key: 'Mn', label: 'Manganese',              unit: 'mg/kg', min: 0,   max: 200,  step: 0.1,  sample: 15    },
  { key: 'B',  label: 'Boron',                  unit: 'mg/kg', min: 0,   max: 10,   step: 0.1,  sample: 0.5   },
];

export const SAMPLE_VALUES = Object.fromEntries(
  NUTRIENT_FIELDS.map(f => [f.key, f.sample])
);

export const EMPTY_NUTRIENTS = Object.fromEntries(
  NUTRIENT_FIELDS.map(f => [f.key, ''])
);
