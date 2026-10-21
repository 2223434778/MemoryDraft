"""Independent RMSNorm-only execution, using the bounded compilation pipeline."""
from scripts.run_asd_compile_pipeline import main


if __name__ == "__main__":
    main(study="rmsnorm", extra_tests=("tests/test_asd_rmsnorm.py",))
