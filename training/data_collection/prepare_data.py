"""Data collection, augmentation, and preparation pipeline.

This module provides tools to:
  1. Expand & validate existing datasets
  2. Apply offline augmentation to the image dataset
  3. Synthesize additional tabular fertility samples via SMOTE
  4. Produce a clean, analysis-ready data split for training

Usage:
    python -m training.data_collection.prepare_data --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import FERTILITY_FEATURE_NAMES, FERTILITY_LABELS
from src.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Image dataset helpers
# ---------------------------------------------------------------------------

class ImageDatasetPreparer:
    """Expand and validate the soil image dataset.

    Provides offline augmentation (brightness, contrast, hue-shift, crop,
    blur) to multiply the effective training set size.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        target_size: Tuple[int, int] = (299, 299),
        random_state: int = 42,
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.rng = np.random.RandomState(random_state)

    # -- validation ---------------------------------------------------------

    def validate_dataset(self) -> dict:
        """Check directory structure and log per-class counts."""
        stats: dict[str, int] = {}
        issues: list[str] = []

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source dir not found: {self.source_dir}")

        for class_dir in sorted(self.source_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            stats[class_dir.name] = len(images)
            if len(images) == 0:
                issues.append(f"Empty class directory: {class_dir.name}")

            # Verify each image is loadable
            for img_path in images:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    issues.append(f"Corrupt image {img_path.name}: {e}")

        logger.info("Image dataset statistics:")
        total = 0
        for cls, count in stats.items():
            logger.info(f"  {cls}: {count} images")
            total += count
        logger.info(f"  Total: {total} images")

        if issues:
            logger.warning(f"Found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        return {"stats": stats, "issues": issues, "total": total}

    # -- augmentation -------------------------------------------------------

    @staticmethod
    def _augment_image(image: Image.Image, rng: np.random.RandomState) -> list[Image.Image]:
        """Generate augmented variants of a single image."""
        augmented: list[Image.Image] = []

        # 1. Brightness variation
        factor = rng.uniform(0.7, 1.3)
        augmented.append(ImageEnhance.Brightness(image).enhance(factor))

        # 2. Contrast variation
        factor = rng.uniform(0.7, 1.3)
        augmented.append(ImageEnhance.Contrast(image).enhance(factor))

        # 3. Color (saturation) variation – simulates hue shift
        factor = rng.uniform(0.8, 1.2)
        augmented.append(ImageEnhance.Color(image).enhance(factor))

        # 4. Random horizontal flip
        if rng.rand() > 0.5:
            augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        # 5. Slight rotation
        angle = rng.uniform(-15, 15)
        augmented.append(image.rotate(angle, fillcolor=(0, 0, 0)))

        # 6. Slight Gaussian blur
        augmented.append(image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5))))

        # 7. Random crop & resize (zoom effect)
        w, h = image.size
        crop_frac = rng.uniform(0.8, 0.95)
        cw, ch = int(w * crop_frac), int(h * crop_frac)
        left = rng.randint(0, w - cw + 1)
        top = rng.randint(0, h - ch + 1)
        cropped = image.crop((left, top, left + cw, top + ch))
        augmented.append(cropped.resize(image.size, Image.LANCZOS))

        return augmented

    def augment_dataset(
        self,
        augmentation_factor: int = 3,
        min_per_class: int = 200,
    ) -> dict:
        """Augment images so each class has at least `min_per_class` images.

        Args:
            augmentation_factor: Max number of augmented copies per image.
            min_per_class: Target minimum images per class.

        Returns:
            Summary statistics.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, dict] = {}

        for class_dir in sorted(self.source_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            out_class = self.output_dir / class_dir.name
            out_class.mkdir(parents=True, exist_ok=True)

            originals = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            n_originals = len(originals)

            # Copy originals
            for img_path in originals:
                dest = out_class / img_path.name
                if not dest.exists():
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(self.target_size, Image.LANCZOS)
                    img.save(dest, quality=95)

            # Augment if needed
            needed = max(0, min_per_class - n_originals)
            augmented_count = 0

            if needed > 0:
                logger.info(
                    f"  {class_dir.name}: {n_originals} originals, "
                    f"generating ~{needed} augmented images"
                )
                while augmented_count < needed:
                    src_img_path = originals[augmented_count % n_originals]
                    img = Image.open(src_img_path).convert("RGB")
                    img = img.resize(self.target_size, Image.LANCZOS)

                    variants = self._augment_image(img, self.rng)
                    for j, variant in enumerate(variants[:augmentation_factor]):
                        if augmented_count >= needed:
                            break
                        fname = f"aug_{augmented_count:04d}_{j}_{src_img_path.stem}.jpg"
                        variant.save(out_class / fname, quality=90)
                        augmented_count += 1

            total = n_originals + augmented_count
            summary[class_dir.name] = {
                "originals": n_originals,
                "augmented": augmented_count,
                "total": total,
            }
            logger.info(f"  {class_dir.name}: {total} total images")

        return summary


# ---------------------------------------------------------------------------
# Tabular dataset helpers
# ---------------------------------------------------------------------------

class TabularDatasetPreparer:
    """Expand, validate, and balance the fertility tabular dataset."""

    def __init__(
        self,
        data_path: Path,
        output_dir: Path,
        random_state: int = 42,
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.random_state = random_state

    def validate_dataset(self) -> dict:
        """Load and validate the CSV dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        required = FERTILITY_FEATURE_NAMES + ["Output"]
        missing = set(required) - set(df.columns)
        issues: list[str] = []
        if missing:
            issues.append(f"Missing columns: {missing}")

        # Check for nulls
        null_counts = df[FERTILITY_FEATURE_NAMES].isnull().sum()
        for col, cnt in null_counts.items():
            if cnt > 0:
                issues.append(f"Column {col} has {cnt} null values")

        # Check for negative values (except pH can be anything)
        for col in FERTILITY_FEATURE_NAMES:
            if col == "pH":
                continue
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                issues.append(f"Column {col} has {negatives} negative values")

        # Class distribution
        distribution: dict[int, int] = {
            int(cast(Any, k)): int(v) for k, v in df["Output"].value_counts().items()
        }
        logger.info("Class distribution:")
        for cls, count in sorted(distribution.items()):
            label = FERTILITY_LABELS.get(cls, f"Unknown({cls})")
            logger.info(f"  {label}: {count}")

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "distribution": distribution,
            "issues": issues,
        }

    def balance_with_smote(
        self,
        output_filename: str = "dataset_balanced.csv",
    ) -> Path:
        """Balance classes using SMOTE oversampling.

        Returns:
            Path to the balanced dataset CSV.
        """

        df = pd.read_csv(self.data_path)
        X = np.asarray(df[FERTILITY_FEATURE_NAMES].values)
        y = np.asarray(df["Output"].values)

        try:
            from imblearn.over_sampling import SMOTE  # type: ignore[import-unresolved]

            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"SMOTE: {len(X)} -> {len(X_resampled)} samples")
        except ImportError:
            logger.warning(
                "imbalanced-learn not installed. Using random oversampling instead."
            )
            X_resampled, y_resampled = self._random_oversample(X, y)

        # Create balanced DataFrame
        df_balanced = pd.DataFrame(X_resampled, columns=FERTILITY_FEATURE_NAMES)
        df_balanced["Output"] = y_resampled

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / output_filename
        df_balanced.to_csv(out_path, index=False)
        logger.info(f"Balanced dataset saved to: {out_path}")

        # Log new distribution
        dist: dict[int, int] = {
            int(cast(Any, k)): int(v)
            for k, v in pd.Series(y_resampled).value_counts().items()
        }
        for cls, count in sorted(dist.items()):
            label = FERTILITY_LABELS.get(cls, f"Unknown({cls})")
            logger.info(f"  {label}: {count}")

        return out_path

    def _random_oversample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple random oversampling fallback."""
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_parts, y_parts = [X], [y]
        rng = np.random.RandomState(self.random_state)

        for cls, cnt in zip(classes, counts):
            if cnt < max_count:
                idx = np.where(y == cls)[0]
                extra = rng.choice(idx, size=max_count - cnt, replace=True)
                X_parts.append(X[extra])
                y_parts.append(y[extra])

        return np.vstack(X_parts), np.concatenate(y_parts)

    def add_engineered_features(
        self,
        df: Optional[pd.DataFrame] = None,
        output_filename: str = "dataset_engineered.csv",
    ) -> Tuple[pd.DataFrame, list[str]]:
        """Add domain-specific engineered features.

        New features:
        - N_P_ratio: Nitrogen to Phosphorus ratio
        - N_K_ratio: Nitrogen to Potassium ratio
        - NPK_total: Sum of N, P, K
        - micro_total: Sum of micronutrients (Zn, Fe, Cu, Mn, B)
        - OC_pH_interaction: Organic Carbon * pH

        Returns:
            Tuple of (DataFrame with new features, list of new feature names).
        """
        if df is None:
            df = pd.read_csv(self.data_path)

        new_features: list[str] = []

        # Ratio features
        df["N_P_ratio"] = df["N"] / (df["P"] + 1e-10)
        new_features.append("N_P_ratio")

        df["N_K_ratio"] = df["N"] / (df["K"] + 1e-10)
        new_features.append("N_K_ratio")

        # Aggregate features
        df["NPK_total"] = df["N"] + df["P"] + df["K"]
        new_features.append("NPK_total")

        df["micro_total"] = df["Zn"] + df["Fe"] + df["Cu"] + df["Mn"] + df["B"]
        new_features.append("micro_total")

        # Interaction features
        df["OC_pH_interaction"] = df["OC"] * df["pH"]
        new_features.append("OC_pH_interaction")

        logger.info(f"Added {len(new_features)} engineered features: {new_features}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / output_filename
        df.to_csv(out_path, index=False)
        logger.info(f"Engineered dataset saved to: {out_path}")

        return df, new_features

    def create_train_val_test_split(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> dict[str, Path]:
        """Create stratified train/val/test splits.

        Returns:
            Dict with paths to each split CSV.
        """
        from sklearn.model_selection import train_test_split

        if df is None:
            df = pd.read_csv(self.data_path)

        y = df["Output"]

        # First split: train+val vs test
        df_trainval, df_test = train_test_split(
            df, test_size=test_size, stratify=y,
            random_state=self.random_state,
        )

        # Second split: train vs val
        adjusted_val = val_size / (1 - test_size)
        df_train, df_val = train_test_split(
            df_trainval, test_size=adjusted_val,
            stratify=df_trainval["Output"],
            random_state=self.random_state,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            p = self.output_dir / f"{name}.csv"
            split_df.to_csv(p, index=False)
            paths[name] = p
            logger.info(f"  {name}: {len(split_df)} samples -> {p}")

        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare & expand datasets for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- validate ---
    val_p = sub.add_parser("validate", help="Validate datasets")
    val_p.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "image_dataset" / "Train")
    val_p.add_argument("--tabular", type=Path, default=PROJECT_ROOT / "data" / "dataset1.csv")

    # -- augment-images ---
    aug_p = sub.add_parser("augment-images", help="Augment image dataset")
    aug_p.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "image_dataset" / "Train")
    aug_p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "image_dataset_augmented" / "Train")
    aug_p.add_argument("--factor", type=int, default=3)
    aug_p.add_argument("--min-per-class", type=int, default=200)

    # -- balance-tabular ---
    bal_p = sub.add_parser("balance-tabular", help="Balance tabular dataset with SMOTE")
    bal_p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "dataset1.csv")
    bal_p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed")

    # -- engineer-features ---
    eng_p = sub.add_parser("engineer-features", help="Add engineered features")
    eng_p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "dataset1.csv")
    eng_p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed")

    # -- split ---
    spl_p = sub.add_parser("split", help="Create train/val/test splits")
    spl_p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "dataset1.csv")
    spl_p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed")
    spl_p.add_argument("--test-size", type=float, default=0.15)
    spl_p.add_argument("--val-size", type=float, default=0.15)

    # -- all ---
    all_p = sub.add_parser("all", help="Run full preparation pipeline")
    all_p.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "image_dataset" / "Train")
    all_p.add_argument("--tabular", type=Path, default=PROJECT_ROOT / "data" / "dataset1.csv")
    all_p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Data Preparation Pipeline")
    logger.info("=" * 60)

    if args.command == "validate":
        img_prep = ImageDatasetPreparer(args.images, args.images)
        img_prep.validate_dataset()
        tab_prep = TabularDatasetPreparer(args.tabular, args.tabular.parent)
        tab_prep.validate_dataset()

    elif args.command == "augment-images":
        prep = ImageDatasetPreparer(args.source, args.output)
        summary = prep.augment_dataset(
            augmentation_factor=args.factor,
            min_per_class=args.min_per_class,
        )
        logger.info(f"Augmentation complete: {json.dumps(summary, indent=2)}")

    elif args.command == "balance-tabular":
        prep = TabularDatasetPreparer(args.data, args.output)
        prep.balance_with_smote()

    elif args.command == "engineer-features":
        prep = TabularDatasetPreparer(args.data, args.output)
        prep.add_engineered_features()

    elif args.command == "split":
        prep = TabularDatasetPreparer(args.data, args.output)
        prep.create_train_val_test_split(
            test_size=args.test_size,
            val_size=args.val_size,
        )

    elif args.command == "all":
        logger.info("--- Step 1: Validate ---")
        img_prep = ImageDatasetPreparer(args.images, args.output / "images")
        img_prep.validate_dataset()

        tab_prep = TabularDatasetPreparer(args.tabular, args.output)
        tab_prep.validate_dataset()

        logger.info("--- Step 2: Augment images ---")
        img_prep.augment_dataset()

        logger.info("--- Step 3: Balance & engineer tabular data ---")
        balanced_path = tab_prep.balance_with_smote()
        df_balanced = pd.read_csv(balanced_path)
        tab_prep.add_engineered_features(df_balanced)

        logger.info("--- Step 4: Create splits ---")
        tab_prep.create_train_val_test_split()

    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
